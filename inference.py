import torch
import re
from config_model import (
    get_tokenizer,
    get_generate_config,
    lora_config,
    bnb_config
)
from constant import PROMTP_ANS_FORMAT, FINETUNED_MODEL, IN_CONTEXT_PROMPT
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import (
    LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
)

def make_ans_prompt(question, choices=None):
    if choices is not None:
        in_context_key = "2"
        choices = f"""### Các lựa chọn\n{choices}
        """
    else:
        in_context_key = "1"
        choices = ""
    
    instruction = question + "\n" + choices
    instruction = instruction.strip()
    
    prompt = PROMTP_ANS_FORMAT.format(
        IN_CONTEXT_PROMPT[in_context_key],
        instruction
    )

    return prompt

def remove_duplicate_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Tách câu dựa trên dấu câu
    seen = set()
    filtered_sentences = []
    
    for sentence in sentences:
        if sentence not in seen:  # Chỉ thêm câu nếu nó chưa xuất hiện trước đó
            filtered_sentences.append(sentence)
            seen.add(sentence)
    
    return " ".join(filtered_sentences)

def inference(tokenizer, model, question, choices, generation_config, device):
    prompt = make_ans_prompt(question, choices)
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    processed_ans = remove_duplicate_sentences(ans)
    processed_ans = processed_ans.split("<|im_start|>assistant")
    return processed_ans[1].strip()

def make_inference(model, tokenizer, generation_config, device):
    # USER INPUT
    question = input("Input your question: ").strip()

    choices = input("Choices (Optional): ").strip()
    choices = choices if choices != "" else None

    print("Generating Answer...")
    ans = inference(tokenizer, model, question, choices, generation_config, device)
    print(ans)

if __name__ == "__main__":
    # Kiểm tra xem có GPU không
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # LOAD MODEL
    print("Loading Model")
    config = PeftConfig.from_pretrained(FINETUNED_MODEL)

    if device == "cuda":
        # Nếu có GPU, dùng quantization 4-bit để tối ưu
        print("GPU detected, loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            quantization_config=bnb_config,  # Quantization cho GPU
            trust_remote_code=True,
            device_map="auto"  # Tự động phân bổ mô hình lên GPU
        )
    else:
        # Nếu không có GPU, dùng FP32 trên CPU
        print("No GPU detected, loading model in FP32 on CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float32,  # FP32 để tránh lỗi trên CPU
            trust_remote_code=True
        )
        model.to("cpu")  # Đảm bảo mô hình chạy trên CPU

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Áp dụng LoRA lên mô hình
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL)

    # Cấu hình sinh văn bản
    generation_config = get_generate_config(tokenizer, model)

    make_inference(model, tokenizer, generation_config, device)