# pip install transformers accelerate langchain bitsandbytes
from typing import List, Optional, Any, Union, Dict
import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LogitsProcessor, StoppingCriteria, StoppingCriteriaList, BitsAndBytesConfig
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- s1 single-pass pieces (transformers) ----------

class BudgetForcingProcessor(LogitsProcessor):
    """
    Single-pass s1-like controller:
      - until min_new_tokens, block EOS and the first token of the end delimiter
      - (optionally) bias ' Wait' token(s) upward to encourage budget-filler
      - (optionally) explicitly replace tail of input_ids with wait tokens
    """
    def __init__(
        self,
        eos_token_id: int,
        delimiter_ids: List[int],        # token ids for "</think>" (no specials)
        wait_ids: List[int],             # token ids for " Wait!" (leading space)
        min_new_tokens: int = 256,       # token budget for 'thinking'
        wait_bias: float = 0.0,          # try 0.5..2.0 if you want a stronger 'Wait' nudge
        start_len: Optional[int] = None, # set at runtime to input length
        force_wait_tokens: bool = False  # if True, explicitly replace tail with wait tokens
    ):
        if eos_token_id is None:
            raise ValueError("eos_token_id cannot be None")
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if wait_bias < 0:
            logger.warning(f"Negative wait_bias ({wait_bias}) may suppress 'Wait' tokens")
            
        self.eos_id = eos_token_id
        self.delim_first = delimiter_ids[0] if delimiter_ids else None
        self.delim_ids = delimiter_ids if delimiter_ids else None
        self.wait_ids = wait_ids or []
        self.min_new = min_new_tokens
        self.wait_bias = wait_bias
        self.start_len = start_len
        self.force_wait_tokens = force_wait_tokens
        
        logger.info(f"BudgetForcingProcessor initialized: min_tokens={min_new_tokens}, "
                   f"wait_bias={wait_bias}, delimiter_ids={delimiter_ids}, wait_ids={wait_ids}, "
                   f"force_wait_tokens={force_wait_tokens}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # how many new tokens have been generated so far
        cur_new = input_ids.shape[1] - (self.start_len or 0)
        
        logger.debug(f"BudgetForcing: current_new_tokens={cur_new}, min_required={self.min_new}")

        if cur_new < self.min_new:
            if cur_new > len(self.delim_ids):
                tail = input_ids[0, -len(self.delim_ids):].tolist()
                is_delimiter_text = tail == self.delim_ids
                if is_delimiter_text:
                    if self.force_wait_tokens and self.wait_ids:
                        # Explicitly force wait tokens by replacing tail of input_ids
                        # This modifies the input_ids in-place to force wait token generation
                        wait_token_id = self.wait_ids[0]  # Use first wait token ID
                        
                        # Replace the last few tokens with wait tokens to force the pattern
                        # We'll replace up to the last 3 tokens (or fewer if sequence is shorter)
                        replace_count = min(3, cur_new)
                        if replace_count > 0:
                            start_idx = input_ids.shape[1] - replace_count
                            for i in range(replace_count):
                                input_ids[0, start_idx + i] = wait_token_id
                            
                            logger.debug(f"Force-replaced last {replace_count} tokens with wait token {wait_token_id}")
                        
                        # Also bias the scores toward wait tokens for next generation
                        for tid in self.wait_ids:
                            if tid < scores.shape[1]:
                                scores[:, tid] = scores[:, tid] + 10.0  # Strong bias when forcing
                    else:
                        # Original behavior: nudge ' Wait' via logit bias
                        wait_tokens_biased = 0
                        for tid in self.wait_ids:
                            if tid < scores.shape[1]:
                                scores[:, tid] = scores[:, tid] + self.wait_bias
                                wait_tokens_biased += 1
                        if wait_tokens_biased > 0 and self.wait_bias > 0:
                            logger.debug(f"Applied wait_bias={self.wait_bias} to {wait_tokens_biased} wait tokens")
        else:
            logger.debug(f"Budget met ({cur_new}>={self.min_new}), allowing natural completion")

        return scores


class DelimiterStop(StoppingCriteria):
    """
    Stop *after* the model has emitted the delimiter (only when budget is already met).
    Simple check: if the last K tokens end with the delimiter ids, we stop.
    """
    def __init__(self, delimiter_ids: List[int], min_new_tokens: int, start_len: int):
        if min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if start_len < 0:
            raise ValueError("start_len must be non-negative")
            
        self.delim = delimiter_ids
        self.min_new = min_new_tokens
        self.start_len = start_len
        self.k = max(len(delimiter_ids), 1)
        
        logger.info(f"DelimiterStop initialized: delimiter_ids={delimiter_ids}, "
                   f"min_new_tokens={min_new_tokens}, start_len={start_len}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_new = input_ids.shape[1] - self.start_len
        
        if cur_new < self.min_new:
            logger.debug(f"DelimiterStop: budget not met ({cur_new}<{self.min_new}), continuing")
            return False
            
        if len(self.delim) == 0:
            logger.debug("DelimiterStop: no delimiter specified, continuing")
            return False
            
        if input_ids.shape[1] < len(self.delim):
            logger.debug("DelimiterStop: sequence too short for delimiter check")
            return False
            
        tail = input_ids[0, -len(self.delim):].tolist()
        should_stop = tail == self.delim
        
        if should_stop:
            logger.info(f"DelimiterStop: Found delimiter sequence {self.delim} at position {cur_new}, stopping")
        else:
            logger.debug(f"DelimiterStop: Tail {tail} != delimiter {self.delim}, continuing")
            
        return should_stop


# ---------- Minimal LangChain wrapper around transformers ----------

class TransformersS1SinglePass(BaseChatModel):
    """
    A LangChain ChatModel that runs a single-pass generation with s1-style budget forcing.
    """
    model_name: str = "local-transformers-s1"
    tokenizer: Any = None
    model: Any = None
    delimiter_text: str = "</think>"
    stopping_text: str = "</answer>"
    delim_ids: List[int] = []
    stop_ids: List[int] = []
    wait_ids: List[int] = []
    min_new_tokens: int = 256
    max_new_tokens: int = 1024
    wait_bias: float = 0.0
    force_wait_tokens: bool = True
    temperature: float = 0.2
    top_p: float = 0.95
    eos_id: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        delimiter_text: str = "</think>",
        stopping_text: str = "</answer>",
        min_new_tokens: int = 256,
        max_new_tokens: int = 1024,
        wait_token_text: str = " Wait!",
        wait_bias: float = 0.0,  # try >0 to push more 'Wait'
        force_wait_tokens: bool = False,  # if True, explicitly replace tail with wait tokens
        temperature: float = 0.2,
        top_p: float = 0.95,
        # Quantization parameters
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
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
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization simultaneously")
        if bnb_4bit_quant_type not in ["fp4", "nf4"]:
            raise ValueError("bnb_4bit_quant_type must be either 'fp4' or 'nf4'")
            
        logger.info(f"Initializing TransformersS1SinglePass with model: {model_id}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logger.info(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
            self._autoprocessor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_id}: {e}")
            raise
            
        # Setup quantization config if requested
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype or torch.float16,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                )
                logger.info(f"Using quantization: 8bit={load_in_8bit}, 4bit={load_in_4bit}, "
                           f"compute_dtype={bnb_4bit_compute_dtype or torch.float16}, "
                           f"quant_type={bnb_4bit_quant_type}")
            except ImportError:
                logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
                raise
        
        try:
            model_kwargs = {
                "torch_dtype": dtype or torch.float32,
                "device_map": "auto" if device is None else None,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                # When using quantization, don't specify device manually
                if device is not None:
                    logger.warning("Ignoring device parameter when using quantization (device_map='auto' is used)")
                    model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            
            if device is not None and quantization_config is None:
                self.model.to(device)
            
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            if quantization_config:
                logger.info(f"Model quantized: 8bit={load_in_8bit}, 4bit={load_in_4bit}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

        # store s1 params
        self.delimiter_text = delimiter_text
        self.stopping_text = stopping_text
        self.delim_ids = self.tokenizer.encode(delimiter_text, add_special_tokens=False)
        self.stop_ids = self.tokenizer.encode(stopping_text, add_special_tokens=False)
        self.wait_ids = self.tokenizer.encode(wait_token_text, add_special_tokens=False)
        self.min_new_tokens = min_new_tokens
        self.max_new_tokens = max_new_tokens
        self.wait_bias = wait_bias
        self.force_wait_tokens = force_wait_tokens
        self.temperature = temperature
        self.top_p = top_p

        # eos token id (required)
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            # try common fallbacks
            fallback_token = self.tokenizer.eos_token or "</s>"
            self.eos_id = self.tokenizer.convert_tokens_to_ids(fallback_token)
            if self.eos_id is None:
                raise ValueError(f"Could not determine EOS token ID for tokenizer. Tried: {fallback_token}")
            logger.warning(f"EOS token ID not found, using fallback: {self.eos_id}")
        
        logger.info(f"S1 configuration: delimiter='{delimiter_text}' (ids={self.delim_ids}), "
                   f"wait_token='{wait_token_text}' (ids={self.wait_ids}), "
                   f"min_tokens={min_new_tokens}, max_tokens={max_new_tokens}, "
                   f"wait_bias={wait_bias}, force_wait_tokens={force_wait_tokens}, "
                   f"temp={temperature}, top_p={top_p}")

    # ---- LangChain interface ----
    @property
    def _llm_type(self) -> str:
        return "transformers-s1-single-pass"

    def _format_msgs(self, messages: List[BaseMessage]) -> str:
        """Simple chat-to-prompt formatter; replace with your chat_template if model uses one."""
        if not messages:
            raise ValueError("messages list cannot be empty")
            
        parts = []
        for i, m in enumerate(messages):
            if isinstance(m, SystemMessage):
                msg_dict = {
                    'role': 'system',
                    'content': [{
                        'type': 'text',
                        'text': m.content
                        }]
                }
                parts.append(msg_dict)
            elif isinstance(m, HumanMessage):
                msg_dict = {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': m.content
                        }
                    ]
                }
                parts.append(msg_dict)
            elif isinstance(m, AIMessage):
                msg_dict = {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'text',
                            'text': m.content
                        }
                    ]
                }
                parts.append(msg_dict)
            else:
                logger.warning(f"Unknown message type at index {i}: {type(m)}")
        
        logger.debug(f"Formatted {len(messages)} messages into prompt of length {len(parts)}")
        return parts

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        start_time = time.time()
        
        try:
            prompt = self._format_msgs(messages)
            enc = self._autoprocessor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.model.device)
            
            input_length = input_ids.shape[1]
            logger.info(f"Starting generation with input length: {input_length} tokens")

            # set up processors
            bf = BudgetForcingProcessor(
                eos_token_id=self.eos_id,
                delimiter_ids=self.delim_ids,
                wait_ids=self.wait_ids,
                min_new_tokens=self.min_new_tokens,
                wait_bias=self.wait_bias,
                start_len=input_length,
                force_wait_tokens=self.force_wait_tokens,
            )

            generation_start = time.time()
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.eos_id,
                    logits_processor=[bf],
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generation_time = time.time() - generation_start
            
            output_length = out.shape[1]
            tokens_generated = output_length - input_length
            
            logger.info(f"Generation completed: {tokens_generated} new tokens in {generation_time:.2f}s "
                       f"({tokens_generated/generation_time:.1f} tokens/sec)")

            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            prompt_decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # return only the assistant continuation (strip the prompt prefix)
            completion = full[len(prompt_decoded):]
            
            total_time = time.time() - start_time
            logger.info(f"Total generation time: {total_time:.2f}s, completion length: {len(completion)} chars")
            
            return AIMessage(content=completion)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    # LangChain entrypoints
    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Main entry point for generating responses."""
        logger.info(f"Invoke called with {len(messages)} messages")
        try:
            result = self._generate(messages, **kwargs)
            logger.info("Invoke completed successfully")
            return result
        except Exception as e:
            logger.error(f"Invoke failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model and configuration."""
        return {
            "model_name": self.model_name,
            "model_device": str(self.model.device),
            "vocab_size": self.tokenizer.vocab_size,
            "delimiter_text": self.delimiter_text,
            "delimiter_ids": self.delim_ids,
            "stopping_text": self.stopping_text,
            "stopping_ids": self.stop_ids,
            "wait_ids": self.wait_ids,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "wait_bias": self.wait_bias,
            "force_wait_tokens": self.force_wait_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "eos_token_id": self.eos_id
        }


# ---------- Example usage: single-pass s1 with a 1-shot in system ----------

def main():
    """Example usage of the TransformersS1SinglePass model."""
    logger.info("Starting example usage")
    
    # 1-shot scaffold that teaches the delimiter and the structure
    system = SystemMessage(
        content=(
            "You are a careful reasoning assistant. You always make sure to check your answer many times.\n"
            "Format your response as:\n"
            "<think> your reasoning here... </think>\n"
            "<answer> your final answer here... </answer>\n"
            "Example\n"
            "Question: What is 15 + 27?\n"
            "Your response:\n"
            "<think>\n"
            "1) 15+20=35\n"
            "2) 35+7=42\n"
            "3) Double-check: 42\n"
            "</think>\n"
            "<answer> 42 </answer>\n"
        )
    )
    question = HumanMessage(content="A train travels at a maximum speed of 60 mph and stops twice for 10 minutes within a 2.5 hour period. It takes 10 minutes to get up to max speed after stopping and 2 minutes to slow down to a stop each time. Remember that the train is still traveling while accelerating to max speed and slowing down to a stop but at a slower speed. How far has it traveled in total?")

    try:
        # Swap in your local/open model ID (e.g., 'Qwen2.5-7B-Instruct', 'Llama-3-8B-Instruct', etc.)
        chat = TransformersS1SinglePass(
            model_id="google/gemma-3-1b-it",   # example; choose a model you have
            delimiter_text="</think>",
            min_new_tokens=768,       # your thinking budget (tokens)
            max_new_tokens=2048,       # safety cap
            wait_token_text=" Wait!",  # leading space important for many tokenizers
            wait_bias=500.0,            # small nudge toward 'Wait' before budget is met
            temperature=0.2,
            top_p=0.95,
            # Quantization options (uncomment to use):
            load_in_4bit=True,       # Enable 4-bit quantization for memory efficiency
            bnb_4bit_compute_dtype=torch.float32,  # Compute dtype for 4-bit
            bnb_4bit_quant_type="nf4",  # Quantization type: "nf4" or "fp4"
        )
        
        # Log model info
        model_info = chat.get_model_info()
        logger.info(f"Model configuration: {model_info}")

        # Use like any LC chat model:
        result = chat.invoke([system, question])
        print("\n" + "="*50)
        print("GENERATED RESPONSE:")
        print("="*50)
        print(result.content)
        print("="*50)
        
        logger.info("Example completed successfully")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    main()
