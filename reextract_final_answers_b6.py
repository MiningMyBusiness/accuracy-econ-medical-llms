#!/usr/bin/env python3
"""
Script to re-extract final answers from B6 style LLM responses.
This script looks for "Final Answer" substring in the final LLM response
and extracts whatever comes after it.
"""

import os
import json
import jsonlines
import argparse
import logging
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_final_answer_from_text(text: str) -> str:
    """
    Extract the final answer from LLM response text by looking for "Final Answer" substring.
    
    Args:
        text: The LLM response text to extract from
        
    Returns:
        The extracted final answer, or the last non-empty line if no "Final Answer" found
    """
    if not isinstance(text, str):
        return str(text)
    
    lines = text.strip().split('\n')
    
    # Find the last instance of "Final Answer" in the entire text (case insensitive)
    text_lower = text.lower()
    last_final_answer_idx = text_lower.rfind('final answer')
    
    if last_final_answer_idx != -1:
        # Extract everything after the last "final answer"
        answer_start = last_final_answer_idx + len('final answer')
        answer = text[answer_start:].strip()
        return answer
    
    # If no explicit "Final Answer:" found, return the last 3 non-empty lines
    non_empty_lines = []
    for line in reversed(lines):
        if line.strip():
            non_empty_lines.append(line.strip())
            if len(non_empty_lines) >= 3:
                break
    
    if non_empty_lines:
        # Reverse to maintain original order and join with newlines
        return '\n'.join(reversed(non_empty_lines))
    
    return text.strip()


def extract_final_answer_from_b6_response(llm_response: Dict[str, Any]) -> str:
    """
    Extract final answer from B6 style LLM response structure.
    
    For B6 responses, the structure is:
    {
        'responses': [list of responses],
        'total_iterations': int,
        'total_words': int,
        'final_response': str
    }
    
    Args:
        llm_response: The LLM response dictionary from B6 style processing
        
    Returns:
        The extracted final answer
    """
    if isinstance(llm_response, str):
        # Simple string response
        return extract_final_answer_from_text(llm_response)
    
    if isinstance(llm_response, dict):
        # Check if it's a B6 style response with final_response
        if 'final_response' in llm_response:
            final_response = llm_response['final_response']
            return extract_final_answer_from_text(final_response)
        
        # Check if it's a B6 style response with responses list
        elif 'responses' in llm_response and isinstance(llm_response['responses'], list):
            responses = llm_response['responses']
            if responses:
                # Use the last response
                final_response = responses[-1]
                return extract_final_answer_from_text(final_response)
        
        # For other dict structures, try to convert to string
        return extract_final_answer_from_text(str(llm_response))
    
    # Fallback
    return str(llm_response)


def process_jsonl_file(input_file: str, output_file: str, baseline_style_filter: Optional[str] = None):
    """
    Process a JSONL file and re-extract final answers.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        baseline_style_filter: If provided, only process entries with this baseline_style
    """
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_count = 0
    updated_count = 0
    
    logger.info(f"Processing {input_file} -> {output_file}")
    if baseline_style_filter:
        logger.info(f"Filtering for baseline_style: {baseline_style_filter}")
    
    with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
        for item in tqdm(reader, desc="Processing entries", unit="entry"):
            processed_count += 1
            
            # Filter by baseline_style if specified
            if baseline_style_filter and item.get('baseline_style') != baseline_style_filter:
                writer.write(item)
                continue
            
            # Extract original data
            original_final_answer = item.get('final_answer', '')
            llm_response = item.get('llm_response')
            
            # Re-extract final answer
            new_final_answer = extract_final_answer_from_b6_response(llm_response)
            
            # Update the item
            item['final_answer'] = new_final_answer
            item['original_final_answer'] = original_final_answer
            item['reextracted'] = True
            
            if new_final_answer != original_final_answer:
                updated_count += 1
                logger.debug(f"Updated {item.get('dataset', '')}:{item.get('qid', '')}: '{original_final_answer}' -> '{new_final_answer}'")
            
            writer.write(item)
    
    logger.info(f"Processed {processed_count} entries, updated {updated_count} final answers")


def main():
    parser = argparse.ArgumentParser(description="Re-extract final answers from B6 style LLM responses")
    parser.add_argument('input_file', type=str, help="Input JSONL file with LLM responses")
    parser.add_argument('output_file', type=str, help="Output JSONL file with re-extracted answers")
    parser.add_argument('--baseline_style', type=str, default=None,
                       help="Filter for specific baseline style (e.g., 'B6')")
    parser.add_argument('--verbose', '-v', action='store_true',
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    process_jsonl_file(args.input_file, args.output_file, args.baseline_style)


if __name__ == "__main__":
    main()
