import os
import torch
import transformers
from constant import MODEL_NAME, R, LORA_ALPHA, TARGET_MODULES, LORA_DROPOUT, BIAS, TASK_TYPE, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, NUM_RETURN_SEQUENCES
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
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

def get_model(tokenizer=None, bnb_config=None, return_base_model=False, lora_config=None):
    tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
    lora_config = lora_config if lora_config is not None else LoraConfig(
        r=R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=BIAS,
        task_type=TASK_TYPE
    )
    try:
        if return_base_model:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                trust_remote_code=True,
                # quantization_config=bnb_config
            )
        else:
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=bnb_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    device_map="auto",
                    trust_remote_code=True
                )
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    return model

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