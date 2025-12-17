import asyncio
import sys
import os
import json
import uuid
import argparse
import jsonlines
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from mirage_benchmark_s1 import main as benchmark_main

# Set up logging for the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_JSON = os.path.join(os.path.dirname(BASE_DIR), 'mirage_dataset_tests', 'benchmark.json')
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Models to test
MODELS = [
    # 'gemma3:270m',
    # 'gemma3:1b', 
    # 'gemma3:4b',
    'amsaravi/medgemma-4b-it:q6'
]

# Baseline styles to test
BASELINE_STYLES = ['B1', 'B2', 'B3', 'B4', 'B6']

# Min words values to test for B6 style (list of integers)
MIN_WORDS_VALUES = [128, 256, 512, 768, 1024]

def generate_run_id():
    """Generate a unique run identifier with datetime and UUID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    uid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{uid}"

def save_config(config: dict, config_path: str):
    """Save run configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

def count_total_questions(benchmark_json: str) -> int:
    """Count total number of questions in the benchmark file."""
    try:
        with open(benchmark_json, 'r') as f:
            data = json.load(f)
        total = 0
        for dataset, qdict in data.items():
            total += len(qdict)
        return total
    except Exception as e:
        logger.warning(f"Failed to count questions in benchmark file: {e}")
        return 0

def count_processed_questions(output_file: str) -> int:
    """Count number of processed questions in an output JSONL file."""
    if not os.path.exists(output_file):
        return 0
    
    try:
        count = 0
        with jsonlines.open(output_file, 'r') as reader:
            for obj in reader:
                if obj.get('dataset') and obj.get('qid'):
                    count += 1
        return count
    except Exception as e:
        logger.warning(f"Failed to count processed questions in {output_file}: {e}")
        return 0

def check_existing_runs(no_rerun: bool = False) -> dict:
    """Check existing config files and output files to determine what has already been run."""
    existing_runs = {}
    
    if not no_rerun:
        return existing_runs
    
    # Get total questions for progress calculation
    total_questions = count_total_questions(BENCHMARK_JSON)
    
    # Check configs directory for existing runs
    if os.path.exists(CONFIG_DIR):
        for config_file in os.listdir(CONFIG_DIR):
            if config_file.startswith('config_') and config_file.endswith('.json'):
                config_path = os.path.join(CONFIG_DIR, config_file)
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    model_name = config.get('model_name')
                    baseline_style = config.get('baseline_style')
                    output_suffix = config.get('output_suffix')
                    status = config.get('status', 'unknown')
                    min_total_words = config.get('min_total_words')
                    
                    if model_name and baseline_style and output_suffix:
                        # Check if output file exists and count processed questions
                        output_file = os.path.join(RESULTS_DIR, f"mirage_s1_results_{output_suffix}.jsonl")
                        processed_count = count_processed_questions(output_file)
                        has_output = processed_count > 0
                        is_complete = processed_count >= total_questions
                        
                        # For B6 style, include min_words in the key
                        if baseline_style == 'B6' and min_total_words is not None:
                            key = (model_name, baseline_style, min_total_words)
                        else:
                            key = (model_name, baseline_style)
                            
                        existing_runs[key] = {
                            'config_path': config_path,
                            'output_file': output_file,
                            'status': status,
                            'has_output': has_output,
                            'processed_count': processed_count,
                            'total_questions': total_questions,
                            'is_complete': is_complete,
                            'output_suffix': output_suffix,
                            'min_total_words': min_total_words
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to read config file {config_file}: {e}")
    
    return existing_runs

async def run_single_benchmark(model: str, baseline_style: str, run_info: dict, current_run: int, total_runs: int, use_local_ollama: bool, semaphore: asyncio.Semaphore = None, min_words: int = None):
    """Run a single benchmark with optional concurrency control."""
    if semaphore:
        async with semaphore:
            return await _execute_benchmark(model, baseline_style, run_info, current_run, total_runs, use_local_ollama, min_words)
    else:
        return await _execute_benchmark(model, baseline_style, run_info, current_run, total_runs, use_local_ollama, min_words)

async def _execute_benchmark(model: str, baseline_style: str, run_info: dict, current_run: int, total_runs: int, use_local_ollama: bool, min_words: int = None):
    """Execute a single benchmark run."""
    run_id = run_info['run_id']
    output_suffix = run_info['output_suffix']
    
    # Configuration for this run
    config = {
        'run_id': run_id,
        'model_name': model,
        'baseline_style': baseline_style,
        'output_suffix': output_suffix,
        'llm_type': 'ollama',
        'benchmark_json': BENCHMARK_JSON,
        'timestamp': datetime.now().isoformat(),
        'max_questions': None,
        'token_cap': 100,
        'context': None,
        'self_consistency_k': 5
    }
    
    # Only add min_total_words for B6 style
    if baseline_style == 'B6' and min_words is not None:
        config['min_total_words'] = min_words
    
    # Save configuration
    config_filename = f"config_{output_suffix}.json"
    config_path = os.path.join(CONFIG_DIR, config_filename)
    save_config(config, config_path)
    
    logger.info(f"Run {current_run}/{total_runs}: Starting benchmark for model '{model}' with baseline '{baseline_style}'")
    logger.info(f"Output suffix: {output_suffix}")
    
    try:
        # Prepare benchmark_main arguments
        benchmark_args = {
            'benchmark_json': BENCHMARK_JSON,
            'output_suffix': output_suffix,
            'model_name': model,
            'llm_type': 'ollama',
            'use_local_ollama': use_local_ollama,
            'baseline_style': baseline_style,
            'max_questions': None,
            'token_cap': 100,
            'context': None,
            'self_consistency_k': 5
        }
        
        # Only add min_total_words for B6 style
        if baseline_style == 'B6' and min_words is not None:
            benchmark_args['min_total_words'] = min_words
        
        await benchmark_main(**benchmark_args)
        logger.info(f"Run {current_run}/{total_runs}: Successfully completed benchmark for {model} - {baseline_style}")
        
        # Update config with completion status
        config['status'] = 'completed'
        config['completed_at'] = datetime.now().isoformat()
        save_config(config, config_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Run {current_run}/{total_runs}: Failed benchmark for {model} - {baseline_style}: {e}")
        
        # Update config with error status
        config['status'] = 'failed'
        config['error'] = str(e)
        config['failed_at'] = datetime.now().isoformat()
        save_config(config, config_path)
        
        return False

async def run_benchmarks(no_rerun: bool = False, concurrent_models: int = 1, use_local_ollama: bool = False):
    """Run benchmarks for all model and baseline style combinations."""
    
    # Create configs and results directories
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check existing runs if no_rerun is enabled
    existing_runs = check_existing_runs(no_rerun)
    if no_rerun and existing_runs:
        logger.info(f"No-rerun mode enabled. Found {len(existing_runs)} existing run configurations.")
        for key, info in existing_runs.items():
            if len(key) == 3:  # B6 with min_words
                model, baseline, min_words = key
                key_display = f"{model} + {baseline} (min_words={min_words})"
            else:  # Regular baseline
                model, baseline = key
                key_display = f"{model} + {baseline}"
                
            if info['is_complete']:
                status_msg = f"complete ({info['processed_count']}/{info['total_questions']} questions)"
            elif info['has_output']:
                status_msg = f"partial ({info['processed_count']}/{info['total_questions']} questions)"
            else:
                status_msg = f"no output (status: {info['status']})"
            logger.info(f"  {key_display}: {status_msg}")
    
    # Calculate total runs considering min_words variations for B6
    total_runs = 0
    for model in MODELS:
        for baseline_style in BASELINE_STYLES:
            if baseline_style == 'B6':
                total_runs += len(MIN_WORDS_VALUES)
            else:
                total_runs += 1
    current_run = 0
    
    # Determine execution mode
    if concurrent_models > 1:
        logger.info(f"Starting concurrent benchmark runs for {len(MODELS)} models and {len(BASELINE_STYLES)} baseline styles")
        logger.info(f"Concurrent models: {concurrent_models}")
        semaphore = asyncio.Semaphore(concurrent_models)
    else:
        logger.info(f"Starting sequential benchmark runs for {len(MODELS)} models and {len(BASELINE_STYLES)} baseline styles")
        semaphore = None
    
    logger.info(f"Total runs planned: {total_runs}")
    
    # Prepare all benchmark tasks
    tasks = []
    run_counter = 0
    
    for model in MODELS:
        for baseline_style in BASELINE_STYLES:
            # For B6 style, iterate through different min_words values
            if baseline_style == 'B6':
                for min_words in MIN_WORDS_VALUES:
                    run_counter += 1
                    
                    # Check if this combination should be skipped in no-rerun mode
                    if no_rerun:
                        key = (model, baseline_style, min_words)
                        if key in existing_runs:
                            existing_info = existing_runs[key]
                            if existing_info['is_complete']:
                                logger.info(f"Run {run_counter}/{total_runs}: Skipping {model} + {baseline_style} (min_words={min_words}) (complete: {existing_info['processed_count']}/{existing_info['total_questions']} questions)")
                                continue
                            elif existing_info['has_output']:
                                logger.info(f"Run {run_counter}/{total_runs}: Resuming {model} + {baseline_style} (min_words={min_words}) (partial: {existing_info['processed_count']}/{existing_info['total_questions']} questions)")
                                # Use existing output_suffix to resume the run
                                output_suffix = existing_info['output_suffix']
                                run_id = existing_info['output_suffix'].split('_')[-1]  # Extract run_id from suffix
                            elif existing_info['status'] == 'completed':
                                logger.info(f"Run {run_counter}/{total_runs}: Skipping {model} + {baseline_style} (min_words={min_words}) (marked as completed in config)")
                                continue
                            else:
                                logger.info(f"Run {run_counter}/{total_runs}: Re-running {model} + {baseline_style} (min_words={min_words}) (status: {existing_info['status']}, no output file)")
                                run_id = generate_run_id()
                                model_clean = model.replace(':', '_').replace('/', '_')
                                output_suffix = f"{model_clean}_{baseline_style}_min{min_words}_{run_id}"
                        else:
                            # New run
                            run_id = generate_run_id()
                            model_clean = model.replace(':', '_').replace('/', '_')
                            output_suffix = f"{model_clean}_{baseline_style}_min{min_words}_{run_id}"
                    else:
                        # Standard mode - always create new run
                        run_id = generate_run_id()
                        model_clean = model.replace(':', '_').replace('/', '_')
                        output_suffix = f"{model_clean}_{baseline_style}_min{min_words}_{run_id}"
                    
                    # Prepare run info
                    run_info = {
                        'run_id': run_id,
                        'output_suffix': output_suffix
                    }
                    
                    # Create task for this benchmark
                    task = run_single_benchmark(
                        model=model,
                        baseline_style=baseline_style,
                        run_info=run_info,
                        current_run=run_counter,
                        total_runs=total_runs,
                        use_local_ollama=use_local_ollama,
                        semaphore=semaphore,
                        min_words=min_words
                    )
                    tasks.append(task)
            else:
                # For non-B6 styles, run normally
                run_counter += 1
                
                # Check if this combination should be skipped in no-rerun mode
                if no_rerun:
                    key = (model, baseline_style)
                    if key in existing_runs:
                        existing_info = existing_runs[key]
                        if existing_info['is_complete']:
                            logger.info(f"Run {run_counter}/{total_runs}: Skipping {model} + {baseline_style} (complete: {existing_info['processed_count']}/{existing_info['total_questions']} questions)")
                            continue
                        elif existing_info['has_output']:
                            logger.info(f"Run {run_counter}/{total_runs}: Resuming {model} + {baseline_style} (partial: {existing_info['processed_count']}/{existing_info['total_questions']} questions)")
                            # Use existing output_suffix to resume the run
                            output_suffix = existing_info['output_suffix']
                            run_id = existing_info['output_suffix'].split('_')[-1]  # Extract run_id from suffix
                        elif existing_info['status'] == 'completed':
                            logger.info(f"Run {run_counter}/{total_runs}: Skipping {model} + {baseline_style} (marked as completed in config)")
                            continue
                        else:
                            logger.info(f"Run {run_counter}/{total_runs}: Re-running {model} + {baseline_style} (status: {existing_info['status']}, no output file)")
                            run_id = generate_run_id()
                            model_clean = model.replace(':', '_').replace('/', '_')
                            output_suffix = f"{model_clean}_{baseline_style}_{run_id}"
                    else:
                        # New run
                        run_id = generate_run_id()
                        model_clean = model.replace(':', '_').replace('/', '_')
                        output_suffix = f"{model_clean}_{baseline_style}_{run_id}"
                else:
                    # Standard mode - always create new run
                    run_id = generate_run_id()
                    model_clean = model.replace(':', '_').replace('/', '_')
                    output_suffix = f"{model_clean}_{baseline_style}_{run_id}"
                
                # Prepare run info
                run_info = {
                    'run_id': run_id,
                    'output_suffix': output_suffix
                }
                
                # Create task for this benchmark (no min_words for non-B6 styles)
                task = run_single_benchmark(
                    model=model,
                    baseline_style=baseline_style,
                    run_info=run_info,
                    current_run=run_counter,
                    total_runs=total_runs,
                    use_local_ollama=use_local_ollama,
                    semaphore=semaphore,
                    min_words=None
                )
                tasks.append(task)
    
    # Execute all tasks
    if tasks:
        if concurrent_models > 1:
            logger.info(f"Executing {len(tasks)} benchmark tasks with up to {concurrent_models} concurrent models")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log summary of results
            successful = sum(1 for r in results if r is True)
            failed = sum(1 for r in results if r is False or isinstance(r, Exception))
            logger.info(f"Concurrent execution completed: {successful} successful, {failed} failed")
        else:
            logger.info(f"Executing {len(tasks)} benchmark tasks sequentially")
            for task in tasks:
                await task
    
    logger.info(f"All benchmark runs completed. Check the configs/ directory for run configurations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks for all model and baseline combinations.")
    parser.add_argument('--no-rerun', action='store_true', 
                       help="Enable no-rerun mode: check existing configs and output files to skip already processed runs")
    parser.add_argument('--concurrent-models', type=int, default=1, choices=[1, 2, 3, 4],
                       help="Number of models to run concurrently (1-4, default: 1 for sequential)")
    parser.add_argument('--min-words', type=int, nargs='+', default=MIN_WORDS_VALUES,
                       help=f"List of min_words values to test for B6 style (default: {MIN_WORDS_VALUES})")
    parser.add_argument('--use-local-ollama', action='store_true',
                          help="Use local ollama client instead of remote API")
    
    args = parser.parse_args()
    
    # Update MIN_WORDS_VALUES with command line argument
    MIN_WORDS_VALUES = args.min_words
    
    if args.no_rerun:
        logger.info("No-rerun mode enabled: will skip runs with existing output files")
    else:
        logger.info("Standard mode: will run all combinations regardless of existing files")
    
    if args.concurrent_models > 1:
        logger.info(f"Concurrent execution mode: up to {args.concurrent_models} models will run simultaneously")
    else:
        logger.info("Sequential execution mode: models will run one at a time")
    
    logger.info(f"Min words values for B6 style: {MIN_WORDS_VALUES}")
    
    asyncio.run(run_benchmarks(no_rerun=args.no_rerun, concurrent_models=args.concurrent_models, use_local_ollama=args.use_local_ollama))
