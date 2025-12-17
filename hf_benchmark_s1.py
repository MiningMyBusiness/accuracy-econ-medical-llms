import os
import json
import time
import logging
from typing import Dict, Any, List
import jsonlines
from tqdm import tqdm
from prompt_builders import prompt_cot, prompt_reflection_once, prompt_rag_only, extract_final_answer, prompt_direct_no_reasoning
import statistics
from collections import Counter
from transformers import pipeline
from homemade_pipeline import HomemadePipeline
import torch
from datasets import load_dataset
import csv
import re
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Helper: load benchmark questions
def load_benchmark_questions(json_path: str) -> List[Dict[str, Any]]:
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
                'answer': qinfo.get('answer', None),
                'context': None  # No context for benchmark.json questions
            })
    logger.info(f"Loaded {len(questions)} questions from benchmark file.")
    return questions

# Helper: load PubMedQA questions from HuggingFace
def load_pubmedqa_questions(split: str = 'train', subset: str = 'pqa_labeled') -> List[Dict[str, Any]]:
    """Load PubMedQA dataset from HuggingFace.
    
    Args:
        split: Dataset split to load (e.g., 'test', 'train', 'validation')
        subset: Subset name (default: 'pqa_labeled')
    
    Returns:
        List of question dictionaries with format matching benchmark.json
    """
    logger.info(f"Loading PubMedQA dataset from HuggingFace: qiaojin/PubMedQA, subset={subset}, split={split}")
    
    try:
        dataset = load_dataset('qiaojin/PubMedQA', subset, split=split)
    except Exception as e:
        logger.error(f"Failed to load PubMedQA dataset: {e}")
        raise
    
    questions = []
    for idx, item in enumerate(dataset):
        # Extract question
        question_text = item['question']
        
        # Extract and concatenate contexts
        # The 'context' field contains 'contexts' which is a dict with 'contexts' key containing list of snippets
        context_data = item.get('context', {})
        if isinstance(context_data, dict):
            context_snippets = context_data.get('contexts', [])
        else:
            context_snippets = []
        
        # Concatenate all context snippets
        concatenated_context = '\n\n'.join(context_snippets) if context_snippets else ''
        
        # Extract answer (final_decision field contains 'yes', 'no', or 'maybe')
        answer = item.get('final_decision', None)
        
        # PubMedQA always has these three options
        options = {
            'A': 'yes',
            'B': 'maybe',
            'C': 'no'
        }
        
        # Use pubmed_id as qid if available, otherwise use index
        qid = item.get('pubid', str(idx))
        
        questions.append({
            'dataset': 'PubMedQA',
            'qid': str(qid),
            'question': question_text,
            'options': options,
            'answer': answer,
            'context': concatenated_context
        })
    
    logger.info(f"Loaded {len(questions)} questions from PubMedQA dataset.")
    return questions

# Helper: load medical exam questions from CSV files
def load_medical_exam_questions(csv_dir: str = None) -> List[Dict[str, Any]]:
    """Load medical exam questions from CSV files in the clinical_benchmarks/medical_exams folder.
    
    Args:
        csv_dir: Directory containing CSV files. If None, uses default path relative to this script.
    
    Returns:
        List of question dictionaries with format matching benchmark.json
    """
    if csv_dir is None:
        # Default path: s1_test_time/clinical_benchmarks/medical_exams
        csv_dir = os.path.join(
            os.path.dirname(__file__),
            'clinical_benchmarks',
            'medical_exams'
        )
    
    logger.info(f"Loading medical exam questions from CSV files in: {csv_dir}")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return []
    
    logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    questions = []
    
    for csv_file in csv_files:
        # Extract dataset name from filename (e.g., 'general_surgery.csv' -> 'general_surgery')
        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        logger.info(f"Loading questions from {dataset_name}.csv")
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader):
                question_text = row['question'].strip()
                answer = row['answer'].strip()
                
                # Extract options from the question text
                # Options are in format: "A. option text\nB. option text\n..."
                options = {}
                
                # Split question into main question and options
                # Options typically start after a newline and follow pattern "A. ", "B. ", etc.
                lines = question_text.split('\n')
                
                main_question_lines = []
                option_lines = []
                in_options = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line starts with an option marker (A., B., C., D.)
                    option_match = re.match(r'^([A-D])\.\s+(.+)$', line)
                    
                    if option_match:
                        in_options = True
                        option_letter = option_match.group(1)
                        option_text = option_match.group(2)
                        options[option_letter] = option_text
                        option_lines.append(line)
                    elif not in_options:
                        main_question_lines.append(line)
                    else:
                        # This is a continuation of the previous option
                        if option_lines:
                            option_lines[-1] += ' ' + line
                            # Update the last option with the continuation
                            last_option_letter = list(options.keys())[-1]
                            options[last_option_letter] += ' ' + line
                
                # Reconstruct the main question without options
                main_question = '\n'.join(main_question_lines).strip()
                
                # Create a unique qid for this question
                qid = f"{dataset_name}_{idx}"
                
                questions.append({
                    'dataset': f'medical_exam_{dataset_name}',
                    'qid': qid,
                    'question': main_question,
                    'options': options,
                    'answer': answer,
                    'context': None  # No context for these questions
                })
        
        logger.info(f"Loaded {sum(1 for q in questions if q['dataset'] == f'medical_exam_{dataset_name}')} questions from {dataset_name}.csv")
    
    logger.info(f"Loaded total of {len(questions)} questions from all medical exam CSV files.")
    return questions

# Helper: build baseline prompts B0-B6 using prompt_builders.py functions
def build_baseline_prompt(
    question: str, 
    options: Dict[str, str] = None, 
    style: str = 'B0',
    context: str = None,
    ctx_list: List[Dict[str, str]] = None,
    token_cap: int = 100
) -> str:
    """Build baseline prompts B0-B6 using existing prompt_builders functions."""
    
    if style == 'B0':  # Direct Answer (no CoT, greedy)
        return prompt_direct_no_reasoning(question, context, ctx_list, options)
    
    elif style == 'B1':  # Standard CoT
        return prompt_cot(question, context, ctx_list, min_lines=8, options=options)
    
    elif style == 'B2':  # Short-CoT with token cap
        # Use CoT but modify to include token cap instruction
        return prompt_cot(question, context, ctx_list, min_lines=0, options=options,
                                 token_cap=token_cap)
    
    elif style == 'B4':  # Reflection-1 (used for first pass)
        return prompt_cot(question, context, ctx_list, min_lines=0, options=options)
    
    elif style == 'B5':  # RAG-only (extractive)
        if not (context or ctx_list):
            logger.warning("B5 (RAG-only) requires context but none provided. Falling back to B0.")
            return build_baseline_prompt(question, options, 'B0')
        return prompt_rag_only(question, ctx_list, context, options)
    
    elif style == 'B6':  # Iterative CoT (used for first pass)
        return prompt_cot(question, context, ctx_list, min_lines=0, options=options)
    
    else:
        logger.warning(f"Unknown baseline style '{style}', defaulting to 'B0'")
        return build_baseline_prompt(question, options, 'B0', context, ctx_list)

def build_reflection_prompt(question: str, options: Dict[str, str], draft_response: str, context: str = None, ctx_list: List[Dict[str, str]] = None) -> str:
    """Build reflection prompt for B4 second pass using prompt_builders."""
    return prompt_reflection_once(question, context, ctx_list, draft_response, options)

def remove_final_answer_line(text: str) -> str:
    """Remove the 'Final Answer:' line or the last line from the response."""
    lines = text.strip().split('\n')
    
    # First, try to find and remove the "Final Answer:" line
    for i, line in enumerate(lines):
        if line.strip().startswith('Final Answer:'):
            return '\n'.join(lines[:i]).strip()
    
    # If no "Final Answer:" line found, remove the last non-empty line
    while lines and not lines[-1].strip():
        lines.pop()
    
    if lines:
        lines.pop()
    
    return '\n'.join(lines).strip()

def build_continuation_prompt(question: str, options: Dict[str, str], previous_reasoning: str, context: str = None, ctx_list: List[Dict[str, str]] = None) -> str:
    """Build continuation prompt for B6 iterative reasoning."""
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    # Format context if provided
    context_text = ""
    if ctx_list:
        from prompt_builders import format_ctx_list
        context_text = f"Use only the passages below.\n\n{format_ctx_list(ctx_list)}\n"
    elif context:
        context_text = f"Context:\n{context}\n\n"
    
    return (
        f"Continue your reasoning from where you left off. Build upon your previous reasoning to reach a final answer.\n\n"
        f"{context_text}"
        f"Question: {question}{options_text}\n\n"
        f"Previous reasoning:\n{previous_reasoning}\nWAIT! Let me check my reasoning to be sure.\n\n"
        f"Continue reasoning:\n"
    )

def count_words(text: str) -> int:
    """Count the number of words in a text."""
    return len(text.split())

def majority_vote(responses: List[str]) -> str:
    """Perform majority vote on a list of responses."""
    # Extract final answers from responses
    answers = [extract_final_answer(resp) for resp in responses]
    # Count occurrences
    counter = Counter(answers)
    # Return most common answer
    return counter.most_common(1)[0][0] if counter else answers[0] if answers else ""

def is_llm_failure(response: str) -> bool:
    """Check if a response indicates an LLM call failure."""
    if not isinstance(response, str):
        return True
    return response.startswith("LLM call failed:") or response.startswith("First pass LLM call failed:") or response.startswith("Second pass LLM call failed:") or "LLM call failed:" in response

# Helper: save result as JSONL
def append_result_jsonl(result: dict, jsonl_path: str):
    logger.debug(f"Appending result for {result.get('dataset', '')}:{result.get('qid', '')} to {jsonl_path}")
    with jsonlines.open(jsonl_path, 'a') as writer:
        writer.write(result)
    logger.info(f"Result written to {jsonl_path} for {result.get('dataset', '')}:{result.get('qid', '')}")

# Main LLM invocation routine
def main(
    benchmark_json: str,
    output_suffix: str,
    results_dir: str,
    model_name: str = None,
    baseline_style: str = 'B0',
    max_questions: int = None,
    token_cap: int = 100,
    context: str = None,
    self_consistency_k: int = 5,
    min_total_words: int = 200,
    max_consecutive_failures: int = 2,
    device: str = "cuda",
    dataset_source: str = 'benchmark',
    pubmedqa_split: str = 'train',
    pubmedqa_subset: str = 'pqa_labeled',
    medical_exam_csv_dir: str = None
):
    os.makedirs(results_dir, exist_ok=True)

    # Determine task based on model name
    if '1b' in model_name:
        task = "text-generation"
    else:
        task = "image-text-to-text"

    if 'mistral' in model_name:
        pipe = HomemadePipeline(model_name)
    else:
        pipe = pipeline(
            task,
            model=model_name,
            device=device,
            torch_dtype=torch.bfloat16
    )

    output_jsonl = os.path.join(results_dir, f"hf_s1_results_{output_suffix}.jsonl")
    logger.info(f"Writing results to output jsonl file: {output_jsonl}")
    
    # Load questions based on dataset source
    if dataset_source == 'pubmedqa':
        questions = load_pubmedqa_questions(split=pubmedqa_split, subset=pubmedqa_subset)
        logger.info(f"Loaded {len(questions)} questions from PubMedQA dataset.")
    elif dataset_source == 'medical_exam':
        questions = load_medical_exam_questions(csv_dir=medical_exam_csv_dir)
        logger.info(f"Loaded {len(questions)} questions from medical exam CSV files.")
    else:  # default to benchmark.json
        questions = load_benchmark_questions(benchmark_json)
        logger.info(f"Loaded {len(questions)} questions from benchmark file.")
    
    if max_questions:
        logger.info(f"Limiting to first {max_questions} questions as requested.")
        questions = questions[:max_questions]

    # Load already processed questions if output file exists
    processed_keys = set()
    if os.path.exists(output_jsonl):
        logger.info(f"Output file {output_jsonl} exists. Loading processed questions to skip.")
        try:
            with jsonlines.open(output_jsonl, 'r') as reader:
                for obj in reader:
                    key = (obj.get('dataset'), obj.get('qid'))
                    processed_keys.add(key)
            logger.info(f"Found {len(processed_keys)} processed questions in output file.")
        except Exception as e:
            logger.error(f"Failed to read output file for processed questions: {e}")

    # Filter out already processed questions
    questions_to_process = [q for q in questions if (q['dataset'], q['qid']) not in processed_keys]
    logger.info(f"{len(questions_to_process)} questions left to process after skipping processed ones.")

    # Initialize failure tracking
    consecutive_failures = 0
    
    for q in tqdm(questions_to_process, desc="Processing questions", unit="q", ncols=80):
        idx = questions.index(q) + 1  # For logging, original index
        logger.info(f"Processing question {idx}/{len(questions)}: {q['dataset']}:{q['qid']}")
        
        # Use question-specific context if available (e.g., from PubMedQA)
        question_context = q.get('context', context)
        
        # Process based on baseline style
        if baseline_style == 'B3':  # Self-Consistency k=5
            responses = []
            prompts = []
            question_has_failure = False
            
            # Generate k responses
            for i in range(self_consistency_k):
                prompt_text = build_baseline_prompt(
                    question=q['question'],
                    options=q['options'],
                    style='B1',  # Use standard CoT for each sample
                    context=question_context,
                    token_cap=token_cap
                )
                prompts.append(prompt_text)
                
                try:
                    if '1b' in model_name:
                        messages = [
                            [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                                },
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": prompt_text}]
                                },
                            ],
                        ]
                        output = pipe(messages, max_new_tokens=1224, temperature=0.2)
                        response = output[0][0]['generated_text'][2]['content']
                    elif 'mistral' in model_name:
                        messages = [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant."
                            },
                            {
                                "role": "user",
                                "content": prompt_text
                            }
                        ]
                        response = pipe(messages, max_new_tokens=1224, temperature=0.2)
                    else:
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt_text}
                                ]
                            }
                        ]
                        output = pipe(text=messages, max_new_tokens=1224, temperature=0.2)
                        response = output[0]["generated_text"][-1]["content"]
                    responses.append(response)
                except Exception as e:
                    logger.error(f"LLM call {i+1}/{self_consistency_k} failed for {q['dataset']}:{q['qid']}: {e}")
                    response = f"LLM call failed: {e}"
                    responses.append(response)
                    question_has_failure = True
            
            # Majority vote
            final_answer = majority_vote(responses)
            llm_response = {
                'responses': responses,
                'majority_vote': final_answer
            }
            prompt_text = prompts[0]  # Store first prompt as representative
            
        elif baseline_style == 'B4':  # Reflection-1
            question_has_failure = False
            
            # First pass
            prompt_text_1 = build_baseline_prompt(
                question=q['question'],
                options=q['options'],
                style='B4',
                context=question_context,
                token_cap=token_cap
            )
            
            try:
                if '1b' in model_name:
                    messages_1 = [
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text_1}]
                            },
                        ],
                    ]
                    output = pipe(messages_1, max_new_tokens=612, temperature=0.2)
                    draft_response = output[0][0]['generated_text'][2]['content']
                elif 'mistral' in model_name:
                    messages_1 = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt_text_1
                        }
                    ]
                    draft_response = pipe(messages_1, max_new_tokens=612, temperature=0.2)
                else:
                    messages_1 = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text_1}
                            ]
                        }
                    ]
                    output = pipe(text=messages_1, max_new_tokens=612, temperature=0.2)
                    draft_response = output[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.error(f"First pass LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                draft_response = f"First pass LLM call failed: {e}"
                question_has_failure = True
            
            # Second pass - reflection
            prompt_text_2 = build_reflection_prompt(
                question=q['question'],
                options=q['options'],
                draft_response=draft_response,
                context=question_context
            )
            
            try:
                if '1b' in model_name:
                    messages_2 = [
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text_2}]
                            },
                        ],
                    ]
                    output = pipe(messages_2, max_new_tokens=612, temperature=0.2)
                    final_response = output[0][0]['generated_text'][2]['content']
                elif 'mistral' in model_name:
                    messages_2 = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt_text_2
                        }
                    ]
                    final_response = pipe(messages_2, max_new_tokens=612, temperature=0.2)
                else:
                    messages_2 = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text_2}
                            ]
                        }
                    ]
                    output = pipe(text=messages_2, max_new_tokens=612, temperature=0.2)
                    final_response = output[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.error(f"Second pass LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                final_response = f"Second pass LLM call failed: {e}"
                question_has_failure = True
            
            llm_response = {
                'draft_response': draft_response,
                'final_response': final_response
            }
            final_answer = extract_final_answer(final_response) if isinstance(final_response, str) else str(final_response)
            prompt_text = f"FIRST PASS:\n{prompt_text_1}\n\nSECOND PASS:\n{prompt_text_2}"
            
        elif baseline_style == 'B6':  # Iterative CoT
            question_has_failure = False
            
            # First pass - initial CoT
            prompt_text_1 = build_baseline_prompt(
                question=q['question'],
                options=q['options'],
                style='B6',
                context=question_context,
                token_cap=token_cap
            )
            
            try:
                if '1b' in model_name:
                    messages_1 = [
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text_1}]
                            },
                        ],
                    ]
                    output = pipe(messages_1, max_new_tokens=612, temperature=0.2)
                    response_1 = output[0][0]['generated_text'][2]['content']
                elif 'mistral' in model_name:
                    messages_1 = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt_text_1
                        }
                    ]
                    response_1 = pipe(messages_1, max_new_tokens=612, temperature=0.2)
                else:
                    messages_1 = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text_1}
                            ]
                        }
                    ]
                    output = pipe(text=messages_1, max_new_tokens=612, temperature=0.2)
                    response_1 = output[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.error(f"First pass LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                response_1 = f"First pass LLM call failed: {e}"
                question_has_failure = True
            
            # Initialize tracking variables
            all_responses = [response_1]
            all_prompts = [prompt_text_1]
            total_words = count_words(response_1) if isinstance(response_1, str) else 0
            current_reasoning = remove_final_answer_line(response_1) if isinstance(response_1, str) else str(response_1)
            
            # Continue iterating until minimum word count is reached
            iteration = 2
            max_iterations = 10  # Safety limit
            
            while total_words < min_total_words and iteration <= max_iterations:
                logger.info(f"B6 iteration {iteration} for {q['dataset']}:{q['qid']} - current words: {total_words}, target: {min_total_words}")
                
                # Build continuation prompt
                continuation_prompt = build_continuation_prompt(
                    question=q['question'],
                    options=q['options'],
                    previous_reasoning=current_reasoning,
                    context=question_context
                )
                
                try:
                    if '1b' in model_name:
                        messages_cont = [
                            [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                                },
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": continuation_prompt}]
                                },
                            ],
                        ]
                        output = pipe(messages_cont, max_new_tokens=612, temperature=0.2)
                        response_cont = output[0][0]['generated_text'][2]['content']
                    elif 'mistral' in model_name:
                        messages_cont = [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant."
                            },
                            {
                                "role": "user",
                                "content": continuation_prompt
                            }
                        ]
                        response_cont = pipe(messages_cont, max_new_tokens=612, temperature=0.2)
                    else:
                        messages_cont = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": continuation_prompt}
                                ]
                            }
                        ]
                        output = pipe(text=messages_cont, max_new_tokens=612, temperature=0.2)
                        response_cont = output[0]["generated_text"][-1]["content"]
                except Exception as e:
                    logger.error(f"Iteration {iteration} LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                    response_cont = f"Iteration {iteration} LLM call failed: {e}"
                    question_has_failure = True
                
                all_responses.append(response_cont)
                all_prompts.append(continuation_prompt)
                
                if isinstance(response_cont, str):
                    total_words += count_words(response_cont)
                    # Update current reasoning by removing final answer line from latest response
                    new_reasoning = remove_final_answer_line(response_cont)
                    current_reasoning = f"{current_reasoning}\n{new_reasoning}"
                
                iteration += 1
            
            # Extract final answer from the last response
            final_response = all_responses[-1]
            final_answer = extract_final_answer(final_response) if isinstance(final_response, str) else str(final_response)
            
            llm_response = {
                'responses': all_responses,
                'total_iterations': len(all_responses),
                'total_words': total_words,
                'final_response': final_response
            }
            
            # Combine all prompts for logging
            prompt_parts = []
            for i, prompt in enumerate(all_prompts, 1):
                prompt_parts.append(f"ITERATION {i}:\n{prompt}")
            prompt_text = "\n\n".join(prompt_parts)
            
        else:  # B0, B1, B2, B5
            question_has_failure = False
            
            prompt_text = build_baseline_prompt(
                question=q['question'],
                options=q['options'],
                style=baseline_style,
                context=question_context,
                token_cap=token_cap
            )
            
            max_tokens = 1224
            if baseline_style == 'B2':
                max_tokens = 256
            try:
                if '1b' in model_name:
                    messages = [
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt_text}]
                            },
                        ],
                    ]
                    output = pipe(messages, max_new_tokens=max_tokens, temperature=0.2)
                    llm_response = output[0][0]['generated_text'][2]['content']
                elif 'mistral' in model_name:
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ]
                    llm_response = pipe(messages, max_new_tokens=max_tokens, temperature=0.2)
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text}
                            ]
                        }
                    ]
                    output = pipe(text=messages, max_new_tokens=max_tokens, temperature=0.2)
                    llm_response = output[0]["generated_text"][-1]["content"]
            except Exception as e:
                logger.error(f"LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                llm_response = f"LLM call failed: {e}"
                question_has_failure = True
            
            # Extract final answer
            final_answer = extract_final_answer(llm_response) if isinstance(llm_response, str) else str(llm_response)
        
        # Update failure tracking
        if question_has_failure:
            consecutive_failures += 1
            logger.warning(f"Question {q['dataset']}:{q['qid']} had failures. Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
            
            if consecutive_failures >= max_consecutive_failures:
                error_msg = f"Too many consecutive failures ({consecutive_failures}). Stopping execution to prevent further issues."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            consecutive_failures = 0  # Reset counter on successful question
        
        result = {
            'dataset': q['dataset'],
            'qid': q['qid'],
            'question': q['question'],
            'options': q['options'],
            'ground_truth': q['answer'],
            'llm_response': llm_response,
            'final_answer': final_answer,
            'baseline_style': baseline_style,
            'prompt_text': prompt_text,
            'context': q.get('context', None)  # Include context in output for PubMedQA
        }
        append_result_jsonl(result, output_jsonl)
        logger.info(f"Finished processing {q['dataset']}:{q['qid']}")

if __name__ == "__main__":
    import argparse
    
    # Default path to mirage benchmark.json
    default_benchmark_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'mirage_dataset_tests', 
        'benchmark.json'
    )
    
    parser = argparse.ArgumentParser(description="Invoke LLM on Mirage benchmark using Hugging Face models.")
    parser.add_argument('--benchmark_json', type=str, default=default_benchmark_path, 
                       help="Path to benchmark.json")
    parser.add_argument('--output_suffix', type=str, required=True, 
                       help="Output suffix for results file")
    parser.add_argument('--results_dir', type=str, default="/mnt/datasetd3mlops-kinematics-pvc/kiranb_area/s1-generation-results", 
                          help="Directory to save results file")
    parser.add_argument('--model_name', type=str, default="google/gemma-3-1b-it", 
                       help="Hugging Face model name (e.g., 'google/gemma-3-1b-it', 'google/gemma-3-4b-it', etc.)")
    parser.add_argument('--baseline_style', type=str, 
                       choices=['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'], 
                       default='B0', help="Baseline style: B0=Direct, B1=CoT, B2=Short-CoT, B3=Self-Consistency, B4=Reflection, B5=RAG-only, B6=Iterative-CoT")
    parser.add_argument('--max_questions', type=int, default=None, 
                       help="Max number of questions to process")
    parser.add_argument('--token_cap', type=int, default=100,
                       help="Token cap for B2 (Short-CoT)")
    parser.add_argument('--context', type=str, default=None,
                       help="Additional context to include in prompts")
    parser.add_argument('--self_consistency_k', type=int, default=5,
                       help="Number of samples for B3 (Self-Consistency)")
    parser.add_argument('--min_total_words', type=int, default=200,
                       help="Minimum total words across all responses for B6 (Iterative-CoT)")
    parser.add_argument('--max_consecutive_failures', type=int, default=10,
                       help="Maximum number of consecutive LLM call failures before stopping execution")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument('--dataset_source', type=str, choices=['benchmark', 'pubmedqa', 'medical_exam'], default='benchmark',
                       help="Data source: 'benchmark' for benchmark.json, 'pubmedqa' for HuggingFace PubMedQA dataset, 'medical_exam' for medical exam CSV files")
    parser.add_argument('--pubmedqa_split', type=str, default='train',
                       help="Split to use for PubMedQA dataset (e.g., 'test', 'train', 'validation')")
    parser.add_argument('--pubmedqa_subset', type=str, default='pqa_labeled',
                       help="Subset to use for PubMedQA dataset (default: 'pqa_labeled')")
    parser.add_argument('--medical_exam_csv_dir', type=str, default=None,
                       help="Directory containing medical exam CSV files (default: s1_test_time/clinical_benchmarks/medical_exams)")
    
    args = parser.parse_args()
    logger.info(f"Dataset source: {args.dataset_source}")
    if args.dataset_source == 'benchmark':
        logger.info(f"Using benchmark json path: {args.benchmark_json}")
    elif args.dataset_source == 'pubmedqa':
        logger.info(f"Using PubMedQA dataset: subset={args.pubmedqa_subset}, split={args.pubmedqa_split}")
    elif args.dataset_source == 'medical_exam':
        csv_dir_msg = args.medical_exam_csv_dir if args.medical_exam_csv_dir else "default (s1_test_time/clinical_benchmarks/medical_exams)"
        logger.info(f"Using medical exam CSV files from: {csv_dir_msg}")
    logger.info(f"Using baseline style: {args.baseline_style}")
    logger.info(f"Using model: {args.model_name} on device: {args.device}")

    main(
        args.benchmark_json,
        args.output_suffix,
        args.results_dir,
        args.model_name,
        args.baseline_style,
        args.max_questions,
        args.token_cap,
        args.context,
        args.self_consistency_k,
        args.min_total_words,
        args.max_consecutive_failures,
        args.device,
        args.dataset_source,
        args.pubmedqa_split,
        args.pubmedqa_subset,
        args.medical_exam_csv_dir
    )
