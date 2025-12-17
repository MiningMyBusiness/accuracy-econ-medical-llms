import os
import json
import asyncio
import time
import logging
from typing import Dict, Any, List
import jsonlines
from pit_vis_extractor.llm_init import call_llm
from llm_utils_async.ollama_client import call_ollama_llm as call_local_ollama_llm
from tqdm import tqdm
from prompt_builders import prompt_cot, prompt_reflection_once, prompt_rag_only, extract_final_answer, prompt_direct_no_reasoning
import statistics
from collections import Counter

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
                'answer': qinfo.get('answer', None)
            })
    logger.info(f"Loaded {len(questions)} questions from benchmark file.")
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
async def main(
    benchmark_json: str,
    output_suffix: str,
    model_name: str = None,
    llm_type: str = 'ollama',
    use_local_ollama: bool = False,
    baseline_style: str = 'B0',
    max_questions: int = None,
    token_cap: int = 100,
    context: str = None,
    self_consistency_k: int = 5,
    min_total_words: int = 200,
    max_consecutive_failures: int = 2
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Determine which LLM function to use
    llm_caller = call_local_ollama_llm if use_local_ollama else call_llm
    output_jsonl = os.path.join(results_dir, f"mirage_s1_results_{output_suffix}.jsonl")
    logger.info(f"Writing results to output jsonl file: {output_jsonl}")
    
    questions = load_benchmark_questions(benchmark_json)
    logger.info(f"Loaded {len(questions)} questions from the dataset.")
    
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
                    context=context,
                    token_cap=token_cap
                )
                prompts.append(prompt_text)
                
                messages = {"content": prompt_text}
                time.sleep(0.5)  # Wait before LLM call
                try:
                    if use_local_ollama:
                        response = llm_caller(messages, model_name=model_name, temperature=0.2, max_tokens=1224)
                    else:
                        response = llm_caller(messages, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=1224)
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
                context=context,
                token_cap=token_cap
            )
            
            messages_1 = {"content": prompt_text_1}
            time.sleep(0.5) # Wait before LLM call
            try:
                if use_local_ollama:
                    draft_response = llm_caller(messages_1, model_name=model_name, temperature=0.2, max_tokens=612)
                else:
                    draft_response = llm_caller(messages_1, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=612)
            except Exception as e:
                logger.error(f"First pass LLM call failed for {q['dataset']}:{q['qid']}: {e}")
                draft_response = f"First pass LLM call failed: {e}"
                question_has_failure = True
            
            # Second pass - reflection
            prompt_text_2 = build_reflection_prompt(
                question=q['question'],
                options=q['options'],
                draft_response=draft_response,
                context=context
            )
            
            messages_2 = {"content": prompt_text_2}
            time.sleep(0.5)  # Wait before LLM call
            try:
                if use_local_ollama:
                    final_response = llm_caller(messages_2, model_name=model_name, temperature=0.2, max_tokens=612)
                else:
                    final_response = llm_caller(messages_2, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=612)
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
                context=context,
                token_cap=token_cap
            )
            
            messages_1 = {"content": prompt_text_1}
            time.sleep(0.5) # Wait before LLM call
            try:
                if use_local_ollama:
                    response_1 = llm_caller(messages_1, model_name=model_name, temperature=0.2, max_tokens=612)
                else:
                    response_1 = llm_caller(messages_1, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=612)
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
                    context=context
                )
                
                messages_cont = {"content": continuation_prompt}
                time.sleep(0.5) # Wait before LLM call
                try:
                    if use_local_ollama:
                        response_cont = llm_caller(messages_cont, model_name=model_name, temperature=0.2, max_tokens=612)
                    else:
                        response_cont = llm_caller(messages_cont, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=612)
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
                context=context,
                token_cap=token_cap
            )
            
            messages = {"content": prompt_text}
            max_tokens = 1224
            if baseline_style == 'B2':
                max_tokens = 256
            time.sleep(0.5) # Wait before LLM call
            try:
                if use_local_ollama:
                    llm_response = llm_caller(messages, model_name=model_name, temperature=0.2, max_tokens=max_tokens)
                else:
                    llm_response = llm_caller(messages, model_name=model_name, llm_type=llm_type, temperature=0.2, max_tokens=max_tokens)
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
            'prompt_text': prompt_text
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
    
    parser = argparse.ArgumentParser(description="Invoke LLM on Mirage benchmark using s1_test_time prompt builders.")
    parser.add_argument('--benchmark_json', type=str, default=default_benchmark_path, 
                       help="Path to benchmark.json")
    parser.add_argument('--output_suffix', type=str, required=True, 
                       help="Output suffix for results file")
    parser.add_argument('--model_name', type=str, default="qwen2.5vl:3b", 
                       help="LLM model name")
    parser.add_argument('--llm_type', type=str, default='ollama', 
                       choices=['openai', 'ollama', 'azure_openai'], help="LLM type")
    parser.add_argument('--use_local_ollama', action='store_true', 
                          help="Use local ollama client instead of remote API")
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
    
    args = parser.parse_args()
    logger.info(f"Using benchmark json path: {args.benchmark_json}")
    logger.info(f"Using baseline style: {args.baseline_style}")
    
    asyncio.run(main(
        args.benchmark_json,
        args.output_suffix,
        args.model_name,
        args.llm_type,
        args.use_local_ollama,
        args.baseline_style,
        args.max_questions,
        args.token_cap,
        args.context,
        args.self_consistency_k,
        args.min_total_words,
        args.max_consecutive_failures
    ))
