#!/usr/bin/env python3
"""
Batch runner for benchmark_mirage_s1_cot_transformer_bf.py
Executes experiments for multiple Gemma models with different token budgets.
"""

import os
import subprocess
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS = [
    # "google/gemma-3-1b-it", 
    "google/gemma-3-4b-it",
    "google/medgemma-4b-it"
]

MIN_NEW_TOKENS_CONFIGS = [128, 256, 512, 768, 1024]
WAIT_BIAS_CONFIGS = [500.0]
FORCE_WAIT_TOKENS_CONFIGS = [True]  # set to True, to replace "Final Answer:" with "Wait" tokens
LOAD_IN_8BIT = True
LOAD_IN_4BIT = False
MAX_CONCURRENT_EXPERIMENTS = 15


def get_model_id_suffix(model_id: str) -> str:
    my_dict = {
        "google/gemma-3-270m-it": "gemma3_270m",
        "google/gemma-3-1b-it": "gemma3_1b",
        "google/gemma-3-4b-it": "gemma3_4b",
        "google/medgemma-4b-it": "medgemma_4b"
    }
    return my_dict.get(model_id, "unknown")

def get_quantization_suffix(load_in_4bit: bool, load_in_8bit: bool) -> str:
    if load_in_4bit:
        return "4bit"
    elif load_in_8bit:
        return "8bit"
    else:
        return "full"

def get_max_new_tokens(min_new_tokens: int) -> int:
    """Calculate max_new_tokens as min_new_tokens + 500"""
    return 2048

def create_output_suffix(model_id: str, min_new_tokens: int, load_in_4bit: bool, load_in_8bit: bool, wait_bias: float, force_wait_tokens: bool) -> str:
    """Create a descriptive output suffix for the experiment"""
    model_suffix = get_model_id_suffix(model_id)
    quantization_suffix = get_quantization_suffix(load_in_4bit, load_in_8bit)
    wait_bias_str = str(wait_bias).replace(".", "dot")
    force_wait_str = "force" if force_wait_tokens else "bias"
    return f"{model_suffix}_{quantization_suffix}_wait_{force_wait_str}_{wait_bias_str}_min_new_tokens_{min_new_tokens}"

def run_experiment(model_id: str, min_new_tokens: int, wait_bias: float, force_wait_tokens: bool, script_path: str, artifacts_mode: bool = False, thread_id: int = 0, start_index: int = None, stop_index: int = None) -> Tuple[bool, str, Dict]:
    """Run a single experiment configuration"""
    max_new_tokens = get_max_new_tokens(min_new_tokens)
    output_suffix = create_output_suffix(model_id, min_new_tokens, LOAD_IN_4BIT, LOAD_IN_8BIT, wait_bias, force_wait_tokens)
    
    logger.info(f"[Thread-{thread_id}] Starting experiment: {model_id} with min_tokens={min_new_tokens}, max_tokens={max_new_tokens}, wait_bias={wait_bias}, force_wait_tokens={force_wait_tokens}")
    
    cmd = [
        "python", script_path,
        "--model_id", model_id,
        "--output_suffix", output_suffix,
        "--min_new_tokens", str(min_new_tokens),
        "--max_new_tokens", str(max_new_tokens),
        "--load_in_8bit",  # Enable 8-bit quantization
        "--wait_bias", str(wait_bias),
        "--temperature", "0.2",
        "--top_p", "0.95"
    ]
    
    if force_wait_tokens:
        cmd.append("--force_wait_tokens")

    if start_index is not None:
        cmd.extend(["--start_index", str(start_index)])
    
    if stop_index is not None:
        cmd.extend(["--stop_index", str(stop_index)])
    
    if artifacts_mode:
        cmd.append("--artifacts_mode")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )
        end_time = time.time()
        duration = end_time - start_time
        
        experiment_info = {
            'model_id': model_id,
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_new_tokens,
            'wait_bias': wait_bias,
            'force_wait_tokens': force_wait_tokens,
            'output_suffix': output_suffix,
            'thread_id': thread_id
        }
        
        if result.returncode == 0:
            logger.info(f"[Thread-{thread_id}] Experiment completed successfully in {duration:.1f}s: {output_suffix}")
            return True, f"Success in {duration:.1f}s", experiment_info
        else:
            logger.error(f"[Thread-{thread_id}] Experiment failed: {output_suffix}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False, f"Failed: {result.stderr[:200]}...", experiment_info
            
    except Exception as e:
        experiment_info = {
            'model_id': model_id,
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_new_tokens,
            'wait_bias': wait_bias,
            'force_wait_tokens': force_wait_tokens,
            'output_suffix': output_suffix,
            'thread_id': thread_id
        }
        logger.error(f"[Thread-{thread_id}] Exception running experiment {output_suffix}: {e}")
        return False, f"Exception: {str(e)}", experiment_info

def save_experiment_log(experiments: List[Dict], log_path: str):
    """Save experiment results to JSON log"""
    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'successful': sum(1 for exp in experiments if exp['success']),
            'failed': sum(1 for exp in experiments if not exp['success']),
            'experiments': experiments
        }, f, indent=2)
    logger.info(f"Experiment log saved to: {log_path}")

def generate_experiment_configs() -> List[Tuple[str, int, float, bool]]:
    """Generate exactly 15 experiment configurations"""
    configs = []
    
    # Generate all possible combinations
    for model_id in MODELS:
        for min_new_tokens in MIN_NEW_TOKENS_CONFIGS:
            for wait_bias in WAIT_BIAS_CONFIGS:
                for force_wait_tokens in FORCE_WAIT_TOKENS_CONFIGS:
                    configs.append((model_id, min_new_tokens, wait_bias, force_wait_tokens))
    
    # Select exactly 15 configurations (prioritize diversity)
    if len(configs) >= 15:
        # Take every nth config to ensure diversity across models and parameters
        step = len(configs) // 15
        selected_configs = []
        for i in range(0, len(configs), max(1, step)):
            if len(selected_configs) < 15:
                selected_configs.append(configs[i])
        
        # If we still need more, fill from remaining configs
        remaining = [c for c in configs if c not in selected_configs]
        while len(selected_configs) < 15 and remaining:
            selected_configs.append(remaining.pop(0))
            
        return selected_configs[:15]
    else:
        return configs

def run_experiment_wrapper(args):
    """Wrapper function for running experiments in threads"""
    model_id, min_new_tokens, wait_bias, force_wait_tokens, script_path, artifacts_mode, thread_id, start_index, stop_index = args
    return run_experiment(model_id, min_new_tokens, wait_bias, force_wait_tokens, script_path, artifacts_mode, thread_id, start_index, stop_index)

def main(artifacts_mode: bool = False, start_index: int = None, stop_index: int = None):
    """Main function to run all experiments concurrently"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_script = os.path.join(script_dir, "benchmark_mirage_s1_cot_transformer_bf.py")
    
    # Verify benchmark script exists
    if not os.path.exists(benchmark_script):
        logger.error(f"Benchmark script not found: {benchmark_script}")
        return
    
    # Create logs directory
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate experiment configurations
    experiment_configs = generate_experiment_configs()
    
    # Generate experiment log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"gemma_batch_experiments_{timestamp}.json")
    
    logger.info("="*80)
    logger.info("STARTING GEMMA BATCH EXPERIMENTS (CONCURRENT)")
    logger.info("="*80)
    logger.info(f"Models: {MODELS}")
    logger.info(f"Min new tokens configs: {MIN_NEW_TOKENS_CONFIGS}")
    logger.info(f"Wait bias configs: {WAIT_BIAS_CONFIGS}")
    logger.info(f"Force wait tokens configs: {FORCE_WAIT_TOKENS_CONFIGS}")
    logger.info(f"Total experiments: {len(experiment_configs)}")
    logger.info(f"Max concurrent: {MAX_CONCURRENT_EXPERIMENTS}")
    if start_index is not None or stop_index is not None:
        logger.info(f"Processing questions from index {start_index} to {stop_index}")
    logger.info(f"Benchmark script: {benchmark_script}")
    logger.info(f"Log file: {log_path}")
    logger.info("="*80)
    
    experiments = []
    overall_start_time = time.time()
    
    # Prepare arguments for each experiment
    experiment_args = []
    for i, (model_id, min_new_tokens, wait_bias, force_wait_tokens) in enumerate(experiment_configs):
        args = (model_id, min_new_tokens, wait_bias, force_wait_tokens, benchmark_script, artifacts_mode, i + 1, start_index, stop_index)
        experiment_args.append(args)
    
    # Run experiments concurrently
    completed_count = 0
    total_experiments = len(experiment_args)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_EXPERIMENTS) as executor:
        # Submit all experiments
        future_to_args = {executor.submit(run_experiment_wrapper, args): args for args in experiment_args}
        
        # Process completed experiments
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            model_id, min_new_tokens, wait_bias, force_wait_tokens, _, _, thread_id, _, _ = args
            
            try:
                success, message, experiment_info = future.result()
                completed_count += 1
                
                experiment_record = {
                    'experiment_id': thread_id,
                    'model_id': model_id,
                    'min_new_tokens': min_new_tokens,
                    'max_new_tokens': get_max_new_tokens(min_new_tokens),
                    'wait_bias': wait_bias,
                    'force_wait_tokens': force_wait_tokens,
                    'quantization': '8bit',
                    'success': success,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'output_suffix': experiment_info['output_suffix']
                }
                
                experiments.append(experiment_record)
                
                # Save intermediate results
                save_experiment_log(experiments, log_path)
                
                if success:
                    logger.info(f"✓ Experiment {completed_count}/{total_experiments} completed: {experiment_info['output_suffix']}")
                else:
                    logger.error(f"✗ Experiment {completed_count}/{total_experiments} failed: {message}")
                    
            except Exception as e:
                completed_count += 1
                logger.error(f"✗ Experiment {completed_count}/{total_experiments} exception: {e}")
                
                # Create a failed experiment record
                experiment_record = {
                    'experiment_id': thread_id,
                    'model_id': model_id,
                    'min_new_tokens': min_new_tokens,
                    'max_new_tokens': get_max_new_tokens(min_new_tokens),
                    'wait_bias': wait_bias,
                    'force_wait_tokens': force_wait_tokens,
                    'quantization': '8bit',
                    'success': False,
                    'message': f"Exception: {str(e)}",
                    'timestamp': datetime.now().isoformat(),
                    'output_suffix': create_output_suffix(model_id, min_new_tokens, LOAD_IN_4BIT, LOAD_IN_8BIT, wait_bias, force_wait_tokens)
                }
                experiments.append(experiment_record)
    
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    
    # Final summary
    successful = sum(1 for exp in experiments if exp['success'])
    failed = len(experiments) - successful
    
    logger.info("\n" + "="*80)
    logger.info("BATCH EXPERIMENTS COMPLETED")
    logger.info("="*80)
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(experiments)*100:.1f}%")
    logger.info(f"Total duration: {total_duration/3600:.1f} hours")
    logger.info(f"Average per experiment: {total_duration/len(experiments)/60:.1f} minutes")
    logger.info(f"Results log: {log_path}")
    logger.info("="*80)
    
    if failed > 0:
        logger.info("\nFailed experiments:")
        for exp in experiments:
            if not exp['success']:
                logger.info(f"  - {exp['model_id']} (min_tokens={exp['min_new_tokens']}, wait_bias={exp.get('wait_bias', 'N/A')}, force_wait={exp.get('force_wait_tokens', 'N/A')}): {exp['message']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run batch experiments for Gemma models with CoT generation"
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=None,
        help="Start index for questions to process."
    )
    parser.add_argument(
        '--stop_index',
        type=int,
        default=None,
        help="Stop index for questions to process."
    )
    parser.add_argument(
        '--artifacts_mode', 
        action='store_true',
        help="Save results to /mnt/artifacts instead of local results directory"
    )
    
    args = parser.parse_args()
    main(
        artifacts_mode=args.artifacts_mode,
        start_index=args.start_index,
        stop_index=args.stop_index
    )
