#!/usr/bin/env python3
"""
Wrapper script for transformer_budgetforcer.py to run against mirage dataset with CoT prompting.
Uses the s1 single-pass generation with budget forcing for chain-of-thought reasoning.
"""

import os
import json
import logging
import argparse
from typing import Dict, Any, List
import jsonlines
from tqdm import tqdm
from langchain_core.messages import SystemMessage, HumanMessage

# Import the transformer budget forcer
from transformer_budgetforcer import TransformersS1SinglePass
from prompt_builders import prompt_cot_s1, extract_final_answer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def append_result_jsonl(result: dict, jsonl_path: str):
    """Save result to JSONL file."""
    logger.debug(f"Appending result for {result.get('dataset', '')}:{result.get('qid', '')} to {jsonl_path}")
    with jsonlines.open(jsonl_path, 'a') as writer:
        writer.write(result)

def main(
    benchmark_json: str,
    model_id: str,
    output_suffix: str,
    min_new_tokens: int = 256,
    max_new_tokens: int = 1024,
    wait_bias: float = 0.5,
    force_wait_tokens: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_questions: int = None,
    start_index: int = None,
    stop_index: int = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    artifacts_mode: bool = False
):
    """Main function to run the benchmark."""
    
    # Setup output directory and file
    if artifacts_mode:
        results_dir = '/mnt/artifacts'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        existing_results_dir = os.path.join(base_dir, 'results')
        logger.info("Using artifacts mode: saving results to /mnt/artifacts")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        existing_results_dir = results_dir
    
    os.makedirs(results_dir, exist_ok=True)
    output_jsonl = os.path.join(results_dir, f"mirage_s1_cot_{output_suffix}.jsonl")
    existing_output_jsonl = os.path.join(existing_results_dir, f"mirage_s1_cot_{output_suffix}.jsonl")
    
    logger.info(f"Writing results to: {output_jsonl}")
    
    # Load questions
    questions = load_benchmark_questions(benchmark_json)

    # Slice questions based on start and stop index if provided
    if start_index is not None or stop_index is not None:
        original_count = len(questions)
        start = start_index or 0
        stop = stop_index or len(questions)
        questions = questions[start:stop]
        logger.info(f"Slicing questions from index {start} to {stop}. Processing {len(questions)} out of {original_count} total questions.")

    if max_questions:
        logger.info(f"Limiting to first {max_questions} questions")
        questions = questions[:max_questions]
    
    # Load already processed questions if output file exists
    processed_keys = set()
    if os.path.exists(existing_output_jsonl):
        logger.info(f"Output file exists. Loading processed questions to skip.")
        try:
            with jsonlines.open(existing_output_jsonl, 'r') as reader:
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
    
    # Initialize the s1 model
    logger.info(f"Initializing TransformersS1SinglePass with model: {model_id}")
    try:
        chat_model = TransformersS1SinglePass(
            model_id=model_id,
            delimiter_text="</think>",
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            wait_token_text=" Wait!",
            wait_bias=wait_bias,
            force_wait_tokens=force_wait_tokens,
            temperature=temperature,
            top_p=top_p,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        
        # Log model info
        model_info = chat_model.get_model_info()
        logger.info(f"Model loaded successfully: {model_info}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
    
    # Process questions
    for q in tqdm(questions_to_process, desc="Processing questions", unit="q", ncols=80):
        logger.info(f"Processing {q['dataset']}:{q['qid']}")
        
        try:
            # Build CoT prompt using prompt_cot function
            prompt_text = prompt_cot_s1(
                question=q['question'],
                options=q['options'],
                min_lines=3  # Require at least 3 reasoning lines
            )
            
            # Create messages for the chat model
            messages = [HumanMessage(content=prompt_text)]
            
            # Generate response using s1 single-pass
            response = chat_model.invoke(messages)
            llm_response = response.content
            
            # Extract final answer
            final_answer = extract_final_answer(llm_response)
            
            logger.info(f"Generated response for {q['dataset']}:{q['qid']}")
            
        except Exception as e:
            logger.error(f"Failed to process {q['dataset']}:{q['qid']}: {e}")
            llm_response = f"Error: {str(e)}"
            final_answer = "Error"
        
        # Save result
        result = {
            'dataset': q['dataset'],
            'qid': q['qid'],
            'question': q['question'],
            'options': q['options'],
            'ground_truth': q['answer'],
            'llm_response': llm_response,
            'final_answer': final_answer,  # Renamed from extracted_answer for consistency
            'baseline_style': 'cot_s1',  # Renamed from prompt_style for consistency
            'prompt_text': prompt_text,  # Added for consistency
            'model_id': model_id,
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_new_tokens,
            'wait_bias': wait_bias,
            'force_wait_tokens': force_wait_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        
        append_result_jsonl(result, output_jsonl)
        logger.info(f"Saved result for {q['dataset']}:{q['qid']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run s1 single-pass CoT generation on Mirage benchmark"
    )
    
    # Required arguments
    parser.add_argument(
        '--model_id', 
        type=str, 
        required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        '--output_suffix', 
        type=str, 
        required=True,
        help="Output suffix for results file"
    )
    
    # Optional arguments
    parser.add_argument(
        '--benchmark_json', 
        type=str, 
        default=None,
        help="Path to benchmark.json (default: ../mirage_dataset_tests/benchmark.json)"
    )
    parser.add_argument(
        '--min_new_tokens', 
        type=int, 
        default=256,
        help="Minimum new tokens for s1 budget (default: 256)"
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=1024,
        help="Maximum new tokens (default: 1024)"
    )
    parser.add_argument(
        '--wait_bias', 
        type=float, 
        default=0.5,
        help="Bias for 'Wait' tokens (default: 0.5)"
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        '--max_questions', 
        type=int, 
        default=None,
        help="Maximum number of questions to process"
    )
    parser.add_argument(
        '--start_index', 
        type=int, 
        default=None,
        help="Start index of questions to process"
    )
    parser.add_argument(
        '--stop_index', 
        type=int, 
        default=None,
        help="Stop index of questions to process"
    )
    parser.add_argument(
        '--load_in_4bit', 
        action='store_true',
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        '--load_in_8bit', 
        action='store_true',
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        '--force_wait_tokens', 
        action='store_true',
        help="Use explicit wait token replacement instead of logit bias"
    )
    parser.add_argument(
        '--artifacts_mode', 
        action='store_true',
        help="Save results to /mnt/artifacts instead of local results directory"
    )
    
    args = parser.parse_args()
    
    # Set default benchmark path if not provided
    if args.benchmark_json is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.benchmark_json = os.path.join(script_dir, '..', 'mirage_dataset_tests', 'benchmark.json')
    
    logger.info(f"Using benchmark file: {args.benchmark_json}")
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Output suffix: {args.output_suffix}")
    logger.info(f"S1 config: min_tokens={args.min_new_tokens}, max_tokens={args.max_new_tokens}, wait_bias={args.wait_bias}, force_wait_tokens={args.force_wait_tokens}")
    if args.start_index is not None or args.stop_index is not None:
        logger.info(f"Processing questions from index {args.start_index} to {args.stop_index}")
    
    main(
        benchmark_json=args.benchmark_json,
        model_id=args.model_id,
        output_suffix=args.output_suffix,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        wait_bias=args.wait_bias,
        force_wait_tokens=args.force_wait_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_questions=args.max_questions,
        start_index=args.start_index,
        stop_index=args.stop_index,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        artifacts_mode=args.artifacts_mode
    )
