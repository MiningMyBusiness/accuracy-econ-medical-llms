#!/usr/bin/env python3
"""
Batched version of benchmark_mirage_s1_cot_transformer_bf.py for faster processing.
Processes multiple questions in parallel batches to improve throughput.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List
import jsonlines
from tqdm import tqdm
from langchain_core.messages import SystemMessage, HumanMessage
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the transformer budget forcer
from transformer_budgetforcer import TransformersS1SinglePass
from prompt_builders import prompt_cot, extract_final_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thread-local storage for model instances
thread_local = threading.local()

def get_model(model_config: Dict[str, Any]) -> TransformersS1SinglePass:
    """Get or create a model instance for the current thread."""
    if not hasattr(thread_local, 'model'):
        logger.info(f"Creating model instance for thread {threading.current_thread().name}")
        thread_local.model = TransformersS1SinglePass(**model_config)
    return thread_local.model

def process_question_batch(questions: List[Dict[str, Any]], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a batch of questions using a single model instance."""
    model = get_model(model_config)
    results = []
    
    for q in questions:
        try:
            # Build CoT prompt
            prompt_text = prompt_cot(
                question=q['question'],
                options=q['options'],
                min_lines=3
            )
            
            # Create messages
            messages = [HumanMessage(content=prompt_text)]
            
            # Generate response
            response = model.invoke(messages)
            llm_response = response.content
            
            # Extract final answer
            final_answer = extract_final_answer(llm_response)
            
            logger.info(f"Processed {q['dataset']}:{q['qid']}")
            
        except Exception as e:
            logger.error(f"Failed to process {q['dataset']}:{q['qid']}: {e}")
            llm_response = f"Error: {str(e)}"
            final_answer = "Error"
        
        # Create result
        result = {
            'dataset': q['dataset'],
            'qid': q['qid'],
            'question': q['question'],
            'options': q['options'],
            'ground_truth': q['answer'],
            'llm_response': llm_response,
            'final_answer': final_answer,
            'baseline_style': 'cot_s1',
            'prompt_text': prompt_text,
            'model_id': model_config['model_id'],
            'min_new_tokens': model_config['min_new_tokens'],
            'max_new_tokens': model_config['max_new_tokens'],
            'wait_bias': model_config['wait_bias'],
            'temperature': model_config['temperature'],
            'top_p': model_config['top_p']
        }
        results.append(result)
    
    return results

def load_benchmark_questions(json_path: str) -> List[Dict[str, Any]]:
    """Load benchmark questions from JSON file."""
    logger.info(f"Loading benchmark questions from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    questions = []
    for dataset, qdict in data.items():
        for qid, qinfo in qdict.items():
            questions.append({
                'dataset': dataset,
                'qid': qid,
                'question': qinfo['question'],
                'options': qinfo.get('options', None),
                'answer': qinfo.get('answer', None)
            })
    
    logger.info(f"Loaded {len(questions)} questions from benchmark file.")
    return questions

def append_results_jsonl(results: List[dict], jsonl_path: str):
    """Save multiple results to JSONL file."""
    with jsonlines.open(jsonl_path, 'a') as writer:
        for result in results:
            writer.write(result)

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main(
    benchmark_json: str,
    model_id: str,
    output_suffix: str,
    min_new_tokens: int = 256,
    max_new_tokens: int = 1024,
    wait_bias: float = 0.5,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_questions: int = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    artifacts_mode: bool = False,
    batch_size: int = 4,
    max_workers: int = 2
):
    """Main function to run the batched benchmark."""
    
    # Setup output directory and file
    if artifacts_mode:
        results_dir = '/mnt/artifacts'
        logger.info("Using artifacts mode: saving results to /mnt/artifacts")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    output_jsonl = os.path.join(results_dir, f"mirage_s1_cot_batched_{output_suffix}.jsonl")
    
    logger.info(f"Writing results to: {output_jsonl}")
    
    # Load questions
    questions = load_benchmark_questions(benchmark_json)
    if max_questions:
        logger.info(f"Limiting to first {max_questions} questions")
        questions = questions[:max_questions]
    
    # Load already processed questions if output file exists
    processed_keys = set()
    if os.path.exists(output_jsonl):
        logger.info(f"Output file exists. Loading processed questions to skip.")
        try:
            with jsonlines.open(output_jsonl, 'r') as reader:
                for obj in reader:
                    key = (obj.get('dataset'), obj.get('qid'))
                    processed_keys.add(key)
            logger.info(f"Found {len(processed_keys)} processed questions.")
        except Exception as e:
            logger.error(f"Failed to read output file: {e}")
    
    # Filter out already processed questions
    questions_to_process = [q for q in questions if (q['dataset'], q['qid']) not in processed_keys]
    logger.info(f"{len(questions_to_process)} questions left to process.")
    
    if not questions_to_process:
        logger.info("No questions to process. Exiting.")
        return
    
    # Model configuration
    model_config = {
        'model_id': model_id,
        'delimiter_text': "Final Answer:",
        'min_new_tokens': min_new_tokens,
        'max_new_tokens': max_new_tokens,
        'wait_token_text': " Wait",
        'wait_bias': wait_bias,
        'temperature': temperature,
        'top_p': top_p,
        'load_in_4bit': load_in_4bit,
        'load_in_8bit': load_in_8bit
    }
    
    # Split questions into batches
    question_batches = chunk_list(questions_to_process, batch_size)
    logger.info(f"Processing {len(question_batches)} batches of size {batch_size} with {max_workers} workers")
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(process_question_batch, batch, model_config): i 
            for i, batch in enumerate(question_batches)
        }
        
        # Process completed batches
        with tqdm(total=len(question_batches), desc="Processing batches", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()
                    append_results_jsonl(results, output_jsonl)
                    logger.info(f"Completed batch {batch_idx + 1}/{len(question_batches)}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batched s1 single-pass CoT generation on Mirage benchmark"
    )
    
    # Required arguments
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--output_suffix', type=str, required=True)
    
    # Optional arguments
    parser.add_argument('--benchmark_json', type=str, default=None)
    parser.add_argument('--min_new_tokens', type=int, default=128)  # Reduced default
    parser.add_argument('--max_new_tokens', type=int, default=512)  # Reduced default
    parser.add_argument('--wait_bias', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_questions', type=int, default=None)
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--artifacts_mode', action='store_true')
    parser.add_argument('--batch_size', type=int, default=4, help="Questions per batch")
    parser.add_argument('--max_workers', type=int, default=2, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Set default benchmark path if not provided
    if args.benchmark_json is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.benchmark_json = os.path.join(script_dir, '..', 'mirage_dataset_tests', 'benchmark.json')
    
    logger.info(f"Batched processing config: batch_size={args.batch_size}, max_workers={args.max_workers}")
    
    main(
        benchmark_json=args.benchmark_json,
        model_id=args.model_id,
        output_suffix=args.output_suffix,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        wait_bias=args.wait_bias,
        temperature=args.temperature,
        top_p=args.top_p,
        max_questions=args.max_questions,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        artifacts_mode=args.artifacts_mode,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
