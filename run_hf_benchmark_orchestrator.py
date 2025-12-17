import os
import argparse
import asyncio
import itertools
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def run_command(semaphore, command):
    """Run a command in a subprocess, using a semaphore to limit concurrency."""
    async with semaphore:
        logger.info(f"Starting run: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logger.info(f"Finished run: {' '.join(command)}")
        else:
            logger.error(f"Failed run: {' '.join(command)}")
            logger.error(f"Stderr:\n{stderr.decode().strip()}")
        return process.returncode

async def main(args):
    """Generate and run benchmark commands based on user-defined parameters."""
    # Path to the script to be executed
    script_path = os.path.join(os.path.dirname(__file__), 'hf_benchmark_s1.py')

    commands = []
    base_combinations = list(itertools.product(args.model_names, args.baseline_styles))

    for model, style in base_combinations:
        runs_for_combination = []
        model_suffix = model.split('/')[-1]
        
        # Determine dataset prefix for output suffix
        if args.dataset_source == 'pubmedqa':
            dataset_prefix = 'pubmedqa'
        elif args.dataset_source == 'medical_exam':
            dataset_prefix = 'medical_exam'
        else:
            dataset_prefix = 'benchmark'

        if style == 'B6':
            # For B6, create a run for each value in min_total_words_list
            for min_words in args.min_total_words_list:
                output_suffix = f"{dataset_prefix}_{model_suffix}_{style}_{min_words}words"
                run_params = {'min_words': min_words, 'output_suffix': output_suffix}
                runs_for_combination.append(run_params)
        else:
            # For other styles, create a single run
            output_suffix = f"{dataset_prefix}_{model_suffix}_{style}"
            runs_for_combination.append({'min_words': None, 'output_suffix': output_suffix})

        for run in runs_for_combination:
            command = [
                'python',
                script_path,
                '--model_name', model,
                '--baseline_style', style,
                '--output_suffix', run['output_suffix'],
                '--results_dir', args.results_dir,
                '--max_consecutive_failures', str(args.max_consecutive_failures),
                '--device', args.device,
                '--dataset_source', args.dataset_source
            ]
            
            # Add dataset-specific arguments
            command.extend(['--dataset_source', args.dataset_source])
            if args.dataset_source == 'pubmedqa':
                command.extend(['--pubmedqa_split', args.pubmedqa_split])
                command.extend(['--pubmedqa_subset', args.pubmedqa_subset])
            elif args.dataset_source == 'medical_exam':
                if args.medical_exam_csv_dir:
                    command.extend(['--medical_exam_csv_dir', args.medical_exam_csv_dir])

            # Add optional arguments if they are provided
            if args.max_questions:
                command.extend(['--max_questions', str(args.max_questions)])
            if style == 'B2':
                command.extend(['--token_cap', str(args.token_cap)])
            if style == 'B3':
                command.extend(['--self_consistency_k', str(args.self_consistency_k)])
            if style == 'B6' and run['min_words'] is not None:
                command.extend(['--min_total_words', str(run['min_words'])])
            if args.context:
                command.extend(['--context', args.context])
            
            commands.append(command)

    logger.info(f"Generated {len(commands)} commands to run.")
    logger.info(f"Running up to {args.max_parallel_runs} commands in parallel.")

    semaphore = asyncio.Semaphore(args.max_parallel_runs)
    tasks = [run_command(semaphore, cmd) for cmd in commands]
    
    results = await asyncio.gather(*tasks)
    
    successful_runs = sum(1 for r in results if r == 0)
    failed_runs = len(results) - successful_runs
    
    logger.info(f"Orchestration complete. Successful runs: {successful_runs}, Failed runs: {failed_runs}")

if __name__ == "__main__":
    # Default path to mirage benchmark.json
    default_benchmark_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'mirage_dataset_tests', 
        'benchmark.json'
    )

    parser = argparse.ArgumentParser(description="Orchestration script for hf_benchmark_s1.py.")

    # Required lists for generating command combinations
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help="List of Hugging Face model names.")
    parser.add_argument('--baseline_styles', type=str, nargs='+', required=True, help="List of baseline styles (e.g., B0, B1)." )
    parser.add_argument('--min_total_words_list', type=int, nargs='+', default=[200], help="List of min total words for B6.")

    # Concurrency control
    parser.add_argument('--max_parallel_runs', type=int, default=4, help="Max number of instances to run simultaneously.")

    # Dataset source selection
    parser.add_argument('--dataset_source', type=str, choices=['benchmark', 'pubmedqa', 'medical_exam'], default='benchmark',
                       help="Data source: 'benchmark' for benchmark.json, 'pubmedqa' for HuggingFace PubMedQA dataset, 'medical_exam' for medical exam CSV files")
    
    # Arguments to pass through to hf_benchmark_s1.py
    parser.add_argument('--benchmark_json', type=str, default=default_benchmark_path, help="Path to benchmark.json (used when dataset_source='benchmark')")
    parser.add_argument('--pubmedqa_split', type=str, default='train', help="Split to use for PubMedQA dataset (e.g., 'test', 'train', 'validation')")
    parser.add_argument('--pubmedqa_subset', type=str, default='pqa_labeled', help="Subset to use for PubMedQA dataset")
    parser.add_argument('--medical_exam_csv_dir', type=str, default=None, help="Directory containing medical exam CSV files (default: s1_test_time/clinical_benchmarks/medical_exams)")
    parser.add_argument('--results_dir', type=str, default="/mnt/datasetd3mlops-kinematics-pvc/kiranb_area/s1-generation-results", help="Directory to save results.")
    parser.add_argument('--max_questions', type=int, help="Max number of questions to process.")
    parser.add_argument('--token_cap', type=int, default=100, help="Token cap for B2 (Short-CoT)")
    parser.add_argument('--context', type=str, help="Additional context for prompts.")
    parser.add_argument('--self_consistency_k', type=int, default=5, help="Number of samples for B3.")
    parser.add_argument('--max_consecutive_failures', type=int, default=10, help="Max consecutive failures before stopping a run.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on (e.g., 'cuda', 'cpu').")

    parsed_args = parser.parse_args()
    
    logger.info(f"Dataset source: {parsed_args.dataset_source}")
    if parsed_args.dataset_source == 'benchmark':
        logger.info(f"Using benchmark json: {parsed_args.benchmark_json}")
    elif parsed_args.dataset_source == 'pubmedqa':
        logger.info(f"Using PubMedQA dataset: subset={parsed_args.pubmedqa_subset}, split={parsed_args.pubmedqa_split}")
    elif parsed_args.dataset_source == 'medical_exam':
        csv_dir_msg = parsed_args.medical_exam_csv_dir if parsed_args.medical_exam_csv_dir else "default (s1_test_time/clinical_benchmarks/medical_exams)"
        logger.info(f"Using medical exam CSV files from: {csv_dir_msg}")
    
    asyncio.run(main(parsed_args))
