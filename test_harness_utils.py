import os, time, math
from typing import List, Tuple
from statistics import mean

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def now_ms():
    return int(time.time() * 1000)

def normalize_ans(s: str) -> str:
    return (s or "").strip().strip(".:").lower()

def extract_final_answer(text: str, delimiter: str = "Final Answer:") -> str:
    # grab text after delimiter, else fallback to last line
    if delimiter in text:
        tail = text.split(delimiter, 1)[1]
    else:
        tail = text
    # take first line/sentence chunk
    tail = tail.strip().splitlines()[0].strip()
    # trim quotes
    if tail.startswith(("'", '"')) and tail.endswith(("'", '"')) and len(tail) > 1:
        tail = tail[1:-1]
    return tail

def ci_95(accs: List[int]) -> Tuple[float, float]:
    # simple Wilson or normal approx; use normal approx for simplicity here
    if not accs:
        return (0.0, 0.0)
    p = mean(accs)
    n = len(accs)
    se = math.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return (p - 1.96 * se, p + 1.96 * se)