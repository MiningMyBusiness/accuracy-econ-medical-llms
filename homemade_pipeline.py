from transformers import LlamaTokenizerFast, MistralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HomemadePipeline:
    def __init__(self, model_name_or_path):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_name_or_path = model_name_or_path
        if model_name_or_path == "mistral-small":
            self.tokenizer, self.model = self.init_mistral_small()
        elif model_name_or_path == "biomistral":
            self.tokenizer, self.model = self.init_biomistral()
        elif model_name_or_path == "mistral7B":
            self.tokenizer, self.model = self.init_mistral7B()
        else:
            raise ValueError(f"Unknown model name or path: {model_name_or_path}")

    def init_mistral_small(self):
        device = "cuda"
        tokenizer = LlamaTokenizerFast.from_pretrained('mistralai/Mistral-Small-Instruct-2409')
        tokenizer.pad_token = tokenizer.eos_token
        model = MistralForCausalLM.from_pretrained('mistralai/Mistral-Small-Instruct-2409',
                                                    torch_dtype=torch.bfloat16).to(device)
        return tokenizer, model

    def generate(self, messages, **kwargs):
        device = "cuda"
        model_input = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        outputs = self.model.generate(model_input, **kwargs)
        dec = self.tokenizer.batch_decode(outputs)
        return dec.split('[/INST]')[-1].strip().split('</s>')[0].strip()

    def init_biomistral(self):
        device = "cuda"
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                     torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        return tokenizer, model

    def init_mistral7B(self):
        device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",
                                                     torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        return tokenizer, model
    
    def __call__(self, messages, **kwargs):
        if self.model_name_or_path == "mistral-small":
            return self.generate(messages, **kwargs)
        elif self.model_name_or_path == "biomistral":
            return self.generate(messages, **kwargs)
        elif self.model_name_or_path == "mistral7B":
            return self.generate(messages, **kwargs)