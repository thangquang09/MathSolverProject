import os
import torch
from constant import (
    MODEL_NAME, R, LORA_ALPHA, TARGET_MODULES, LORA_DROPOUT, BIAS, TASK_TYPE, 
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, NUM_RETURN_SEQUENCES
)
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import (
    LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=BIAS,
    task_type=TASK_TYPE
)

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )

def get_generate_config(tokenizer, model):
    generation_config = model.generation_config
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.temperature = TEMPERATURE
    generation_config.top_p = TOP_P
    generation_config.num_return_sequences = NUM_RETURN_SEQUENCES
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    return generation_config

if __name__ == "__main__":
    pass