# pip install vllm "transformers>=4.41" langchain langchain-core langchain-community

from typing import List, Any, Optional, Union, Dict
import logging
import time
from vllm import LLM, SamplingParams
import torch
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------- vLLM logits processor (single-pass budget forcing) --------

class S1BudgetForcerVLLM:
    """
    vLLM logits-processor: until min_new_tokens is reached,
    - mask EOS
    - mask the first token of the end-delimiter ("Final")
    - (optional) add a small positive bias to ' Wait'
    Signature must accept (generated_tokens, logits) and may accept prompt_tokens first.
    """
    def __init__(self, eos_token_id: int, delimiter_first_id: Optional[int],
                 wait_token_ids: List[int], min_new_tokens: int, wait_bias: float):
        if eos_token_id is None:
            raise ValueError("eos_token_id cannot be None")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if wait_bias < 0:
            logger.warning(f"Negative wait_bias ({wait_bias}) may suppress 'Wait' tokens")
            
        self.eos_id = eos_token_id
        self.delim_first = delimiter_first_id
        self.wait_ids = wait_token_ids or []
        self.min_new = min_new_tokens
        self.wait_bias = wait_bias
        self._start_len = None  # set at runtime per request
        
        logger.info(f"S1BudgetForcerVLLM initialized: min_tokens={min_new_tokens}, "
                   f"wait_bias={wait_bias}, delimiter_first_id={delimiter_first_id}, "
                   f"wait_token_ids={wait_token_ids}")

    def set_start_len(self, start_len: int):
        """Set the starting length for this generation request."""
        if start_len < 0:
            raise ValueError("start_len must be non-negative")
        self._start_len = start_len
        logger.debug(f"Set start_len to {start_len}")

    # vLLM calls this as either (generated_tokens, logits) or (prompt_tokens, generated_tokens, logits)
    def __call__(self, *args):
        try:
            if len(args) == 2:
                generated_tokens, logits = args
            elif len(args) == 3:
                # (prompt_tokens, generated_tokens, logits)
                _, generated_tokens, logits = args
            else:
                raise ValueError(f"Expected 2 or 3 arguments, got {len(args)}")

            # generated_tokens is a List[int] for the single sequence in this request
            cur_new = len(generated_tokens)
            logger.debug(f"S1BudgetForcer: current_new_tokens={cur_new}, min_required={self.min_new}")
            
            if cur_new < self.min_new:
                # logits is a 1D tensor for next-token logits
                vocab_size = logits.shape[0]
                
                # Block EOS token
                if self.eos_id is not None and self.eos_id < vocab_size:
                    logits[self.eos_id] = -float("inf")
                    logger.debug(f"Blocked EOS token (id={self.eos_id}) at step {cur_new}")
                
                # Block delimiter first token
                if self.delim_first is not None and self.delim_first < vocab_size:
                    logits[self.delim_first] = -float("inf")
                    logger.debug(f"Blocked delimiter token (id={self.delim_first}) at step {cur_new}")
                
                # Apply wait bias
                wait_tokens_biased = 0
                if self.wait_bias != 0.0:
                    for tid in self.wait_ids:
                        if tid < vocab_size:
                            logits[tid] = logits[tid] + self.wait_bias
                            wait_tokens_biased += 1
                
                if wait_tokens_biased > 0 and self.wait_bias > 0:
                    logger.debug(f"Applied wait_bias={self.wait_bias} to {wait_tokens_biased} wait tokens")
            else:
                logger.debug(f"Budget met ({cur_new}>={self.min_new}), allowing natural completion")
                
            return logits
            
        except Exception as e:
            logger.error(f"Error in S1BudgetForcerVLLM.__call__: {e}")
            raise

# -------- Minimal LangChain chat wrapper around vLLM --------

class VLLMS1SinglePass(BaseChatModel):
    model_name: str = "vllm-s1-single-pass"

    def __init__(
        self,
        model_id: str,
        delimiter_text: str = "Final Answer:",
        min_new_tokens: int = 256,
        max_new_tokens: int = 768,
        wait_token_text: str = " Wait",
        wait_bias: float = 0.3,
        temperature: float = 0.2,
        top_p: float = 0.95,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        super().__init__()
        
        # Validate parameters
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if max_new_tokens <= min_new_tokens:
            raise ValueError("max_new_tokens must be greater than min_new_tokens")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        if not (0.0 < gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")
            
        logger.info(f"Initializing VLLMS1SinglePass with model: {model_id}")
        logger.info(f"vLLM config: tensor_parallel_size={tensor_parallel_size}, "
                   f"gpu_memory_utilization={gpu_memory_utilization}")
        
        try:
            # Start vLLM engine
            self.llm = LLM(
                model=model_id,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            logger.info("vLLM engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine for {model_id}: {e}")
            raise
            
        try:
            self.tokenizer = self.llm.get_tokenizer()
            logger.info(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
        except Exception as e:
            logger.error(f"Failed to get tokenizer: {e}")
            raise
            
        # Store configuration
        self.model_id = model_id
        self.delimiter_text = delimiter_text
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.wait_token_text = wait_token_text
        self.wait_bias = wait_bias
        self.temperature = temperature
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        # Token ids
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            # Try common fallbacks
            fallback_token = self.tokenizer.eos_token or "</s>"
            self.eos_id = self.tokenizer.convert_tokens_to_ids(fallback_token)
            if self.eos_id is None:
                raise ValueError(f"Could not determine EOS token ID for tokenizer. Tried: {fallback_token}")
            logger.warning(f"EOS token ID not found, using fallback: {self.eos_id}")
            
        delim_ids = self.tokenizer.encode(self.delimiter_text, add_special_tokens=False)
        self.delim_first_id = delim_ids[0] if delim_ids else None
        self.wait_ids = self.tokenizer.encode(self.wait_token_text, add_special_tokens=False)
        
        logger.info(f"S1 configuration: delimiter='{delimiter_text}' (first_id={self.delim_first_id}), "
                   f"wait_token='{wait_token_text}' (ids={self.wait_ids}), "
                   f"min_tokens={min_new_tokens}, max_tokens={max_new_tokens}, "
                   f"wait_bias={wait_bias}, temp={temperature}, top_p={top_p}")

    @property
    def _llm_type(self) -> str:
        return "vllm-s1-single-pass"

    def _format_msgs(self, messages: List[BaseMessage]) -> str:
        """Simple chat scaffold; swap to tokenizer.apply_chat_template(...) if your model needs it."""
        if not messages:
            raise ValueError("messages list cannot be empty")
            
        parts = []
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                parts.append(f"[System]\n{m.content}\n")
            elif isinstance(m, HumanMessage):
                parts.append(f"[User]\n{m.content}\n")
            elif isinstance(m, AIMessage):
                parts.append(f"[Assistant]\n{m.content}\n")
            else:
                logger.warning(f"Unknown message type at index {i}: {type(m)}")
                
        parts.append("[Assistant]\n")
        formatted = "\n".join(parts)
        
        logger.debug(f"Formatted {len(messages)} messages into prompt of length {len(formatted)}")
        return formatted

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Main entry point for generating responses."""
        start_time = time.time()
        logger.info(f"Invoke called with {len(messages)} messages")
        
        try:
            prompt = self._format_msgs(messages)
            
            # Calculate input length for metrics
            input_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            input_length = len(input_tokens)
            logger.info(f"Starting generation with input length: {input_length} tokens")
            
            # Build our processor instance for this request
            bf = S1BudgetForcerVLLM(
                eos_token_id=self.eos_id,
                delimiter_first_id=self.delim_first_id,
                wait_token_ids=self.wait_ids,
                min_new_tokens=self.min_new_tokens,
                wait_bias=self.wait_bias,
            )
            
            # vLLM's SamplingParams takes python-callable logits processors
            sampling = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                stop=None,  # we rely on the delimiter mask; add stop=[self.delimiter_text] if you want to stop on it
                logits_processors=[bf],
            )
            
            logger.debug(f"SamplingParams: temp={self.temperature}, top_p={self.top_p}, "
                        f"max_tokens={self.max_new_tokens}")

            # IMPORTANT: we need the input length to compute "new tokens".
            # vLLM gives us only generated tokens, so we treat its length directly as "cur_new".
            # (If you adopt a chat template that inserts system tokens, this remains correct: we only count generated tokens.)

            # Generate (single-pass)
            generation_start = time.time()
            out = self.llm.generate([prompt], sampling_params=sampling)
            generation_time = time.time() - generation_start
            
            if not out or not out[0].outputs:
                raise RuntimeError("vLLM generation returned empty results")
                
            text = out[0].outputs[0].text
            
            # Calculate generation metrics
            output_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens_generated = len(output_tokens)
            
            logger.info(f"Generation completed: {tokens_generated} new tokens in {generation_time:.2f}s "
                       f"({tokens_generated/generation_time:.1f} tokens/sec)")
            
            total_time = time.time() - start_time
            logger.info(f"Total generation time: {total_time:.2f}s, completion length: {len(text)} chars")
            logger.info("Invoke completed successfully")
            
            return AIMessage(content=text)
            
        except Exception as e:
            logger.error(f"Invoke failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model and configuration."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "vocab_size": self.tokenizer.vocab_size,
            "delimiter_text": self.delimiter_text,
            "delimiter_first_id": self.delim_first_id,
            "wait_token_text": self.wait_token_text,
            "wait_ids": self.wait_ids,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "wait_bias": self.wait_bias,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "eos_token_id": self.eos_id
        }

# -------- Example usage with a 1-shot in the system prompt --------

def main():
    """Example usage of the VLLMS1SinglePass model."""
    logger.info("Starting vLLM example usage")
    
    system = SystemMessage(
        content=(
            "You are a careful reasoning assistant.\n"
            "Use this format:\n"
            "Reasoning: <step-by-step reasoning>\n"
            "Final Answer: <concise answer>\n\n"
            "Example\n"
            "Question: What is 15 + 27?\n"
            "Reasoning: 15+20=35; 35+7=42. Double-check: 42.\n"
            "Final Answer: 42\n\n"
            "When you feel ready to end but might be missing steps, write 'Wait' and continue reasoning before 'Final Answer:'."
        )
    )
    question = HumanMessage(content="A train travels 60 mph for 2.5 hours. How far does it go?")

    try:
        chat = VLLMS1SinglePass(
            model_id="Qwen/Qwen2.5-7B-Instruct",   # choose any vLLM-supported HF model id
            delimiter_text="Final Answer:",
            min_new_tokens=256,
            max_new_tokens=768,
            wait_token_text=" Wait",
            wait_bias=0.3,
            temperature=0.2,
            top_p=0.95,
        )
        
        # Log model info
        model_info = chat.get_model_info()
        logger.info(f"Model configuration: {model_info}")
        
        result = chat.invoke([system, question])
        print("\n" + "="*50)
        print("GENERATED RESPONSE:")
        print("="*50)
        print(result.content)
        print("="*50)
        
        logger.info("vLLM example completed successfully")
        
    except Exception as e:
        logger.error(f"vLLM example failed: {e}")
        raise

if __name__ == "__main__":
    main()
