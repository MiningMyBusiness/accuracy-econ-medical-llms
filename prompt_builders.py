
from typing import List, Optional, Dict, Tuple

ONE_SHOT_SCaffold = (
    "You are a careful reasoning assistant.\n"
    "Respond in this format:\n"
    "Reasoning:\n"
    "  1) <step 1>\n"
    "  2) <step 2>\n"
    "  ...\n"
    "Final Answer: <concise answer>\n"
    "<answer-complete>\n\n"
    "Example\n"
    "Question: What is 15 + 27?\n"
    "Reasoning:\n"
    "  1) 15+20=35\n"
    "  2) 35+7=42\n"
    "  3) Double-check: 42\n"
    "Final Answer: 42\n"
    "<answer-complete>\n\n"
)

one_shot_scaffold_s1 = (
    "You are a careful reasoning assistant.\n"
    "Respond in this format:\n"
    "<think> your reasoning here... </think>\n"
    "<answer> your final answer here... </answer>\n"
    "# Example\n"
    "Question:\nWhat is 15 + 27?\n"
    "Your response:\n"
    "<think>\n1) 15+20=35\n2) 35+7=42\n3) Double-check: 42\n</think>\n"
    "<answer> 42 </answer>\n"
)

def format_ctx_list(ctx_list: List[Dict[str, str]]) -> str:
    # Present numbered contexts with IDs; we’ll later check which ID was cited.
    lines = []
    for i, c in enumerate(ctx_list, 1):
        cid = c.get("id", f"C{i}")
        txt = c["text"].strip()
        lines.append(f"[{cid}]\n{txt}\n")
    return "\n".join(lines)

def prompt_direct(question: str, context: Optional[str] = None, ctx_list: Optional[List[Dict[str,str]]] = None, options: Optional[Dict[str, str]] = None) -> str:
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    if ctx_list:
        return (
            f"{ONE_SHOT_SCaffold}\n"
            f"Use only the passages below.\n\n"
            f"{format_ctx_list(ctx_list)}\n"
            f"Question: {question}{options_text}\n"
            f"Final Answer:"
        )
    if context:
        return (
            f"{ONE_SHOT_SCaffold}\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}{options_text}\n"
            f"Final Answer:"
        )
    return (
        f"{ONE_SHOT_SCaffold}\n"
        f"Question: {question}{options_text}\n"
        f"Final Answer:"
    )

def prompt_direct_no_reasoning(question: str, context: Optional[str] = None, ctx_list: Optional[List[Dict[str,str]]] = None, options: Optional[Dict[str, str]] = None) -> str:
    """B0 style: Direct answer with no reasoning required."""
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    # Format context if provided
    context_text = ""
    if ctx_list:
        context_text = f"Use only the passages below.\n\n{format_ctx_list(ctx_list)}\n"
    elif context:
        context_text = f"Context:\n{context}\n\n"
    
    return (
        f"{context_text}"
        f"Question: {question}{options_text}\n"
        f"Answer in one sentence. No reasoning.\n"
        f"Final Answer:"
    )

def prompt_cot(question: str, context: Optional[str] = None, ctx_list: Optional[List[Dict[str,str]]] = None, min_lines: int = 0, options: Optional[Dict[str, str]] = None,
               token_cap: Optional[int] = None) -> str:
    rule = ""
    if min_lines > 0:
        rule = f"\nRules:\n- Do NOT write 'Final Answer:' until you have at least {min_lines} numbered reasoning lines."
    if token_cap and min_lines == 0:
        rule += f"\nRules:\n- Keep your reasoning brief (MAX {token_cap} words)."
    
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    if ctx_list:
        return (
            f"{ONE_SHOT_SCaffold}{rule}\n"
            f"Use relevant information from the passages below.\n\n"
            f"{format_ctx_list(ctx_list)}\n"
            f"Question: {question}{options_text}\n"
            f"Reasoning:\n"
        )
    if context:
        return (
            f"{ONE_SHOT_SCaffold}{rule}\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}{options_text}\n"
            f"Reasoning:\n"
        )
    return (
        f"{ONE_SHOT_SCaffold}{rule}\n"
        f"Question: {question}{options_text}\n"
        f"Reasoning:\n"
    )

def prompt_cot_s1(question: str, min_lines: int = 0, options: Optional[Dict[str, str]] = None) -> str:
    rule = ""
    if min_lines > 0:
        rule = f"\nRules:\n- Do NOT write </think> until you have at least {min_lines} numbered reasoning lines."

    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    return (
        f"{one_shot_scaffold_s1}{rule}\n"
        f"Question:\n{question}{options_text}\n"
        f"Your response:\n"
    )

def prompt_reflection_once(question: str, context: Optional[str], ctx_list: Optional[List[Dict[str,str]]], draft: str, options: Optional[Dict[str, str]] = None) -> str:
    # Second pass: ask to critique the draft reasoning and then answer.
    base = "You wrote the following draft reasoning and answer. Critique for mistakes/omissions, correct them, and then provide a final answer."
    sources = ""
    if ctx_list:
        sources = f"Use only the passages below.\n\n{format_ctx_list(ctx_list)}\n"
    elif context:
        sources = f"Context:\n{context}\n\n"
    
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    return (
        f"{base}\n\n"
        f"{sources}"
        f"Question: {question}{options_text}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Now provide:\n"
        f"Reasoning: <your corrected reasoning>\n"
        f"Final Answer:"
    )

def prompt_rag_only(question: str, ctx_list: Optional[List[Dict[str,str]]], context: Optional[str], options: Optional[Dict[str, str]] = None) -> str:
    # Evidence-first style to test extractive behavior
    # Format options if provided
    options_text = ""
    if options:
        opt_lines = [f"{k}. {v}" for k, v in options.items()]
        options_text = "\nOptions:\n" + "\n".join(opt_lines) + "\n"
    
    if ctx_list:
        return (
            "Answer strictly from the passages. First copy the single most relevant supporting sentence in quotes, "
            "then provide Final Answer.\n\n"
            f"{format_ctx_list(ctx_list)}\n"
            f"Question: {question}{options_text}\n"
            f'Evidence: "'
        )
    if context:
        return (
            "Answer strictly from the context. First copy the single most relevant supporting sentence in quotes, "
            "then provide Final Answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}{options_text}\n"
            f'Evidence: "'
        )
    # No context = fall back to direct (not ideal but keeps harness robust)
    return prompt_direct(question, None, None, options)

def parse_evidence_and_answer(text: str) -> Tuple[str, str]:
    # Very lightweight parser: look for a closing quote then a 'Final Answer:' line
    ev = ""
    ans = ""
    if '"' in text:
        first_close = text.find('"', 1)
        if first_close != -1:
            ev = text[0:first_close+1]
            rest = text[first_close+1:]
            # find 'Final Answer:'
            if "Final Answer:" in rest:
                ans = extract_final_answer(rest)
    if not ans:
        ans = extract_final_answer(text)
    return ev.strip(), ans.strip()

def extract_final_answer(text: str) -> str:
    """Extract the final answer from LLM response text."""
    lines = text.strip().split('\n')
    for line in lines:
        if line.strip().startswith('Final Answer:'):
            return line.replace('Final Answer:', '').strip()
    # If no explicit "Final Answer:" found, return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return text.strip()