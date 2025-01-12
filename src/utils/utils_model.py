import os
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizer, PreTrainedModel


def get_model(config):
    model_name = config["General"].get("model_name") # llama2 or llama3
    model_name = config["MODELS"].get(model_name) # meta-llama/Llama-2-7b-chat-hf or meta-llama/Meta-Llama-3-8B-Instruct
    dtype = config["General"].get("dtype") 
    folder_model = os.path.join(config["Settings"].get("path"), config["General"].get("folder_model"))

    model, tokenizer = load_model(
        model_name=model_name, 
        folder_model=folder_model, 
        dtype=dtype
        )

    return model, tokenizer

def download_models(models, folder_model):
    for model_name in models.values():
        # triggers download of the models
        AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir = folder_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        AutoTokenizer.from_pretrained(
            model_name,
            cache_dir = folder_model)

def load_model(model_name: str, folder_model: str, dtype) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    torch_dtype = torch.float32
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="mps", #auto
        torch_dtype=torch_dtype,
        cache_dir = folder_model

    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir = folder_model
        )
    return model, tokenizer

def generate(
    model: PreTrainedModel,
    inputs: BatchEncoding,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
) -> str:
    output = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_seq_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
    )