MODEL_NAME = "vilm/vinallama-7b-chat" # model name
FINETUNED_MODEL = "thangquang09/vinallama_math_solver_7B"

PROMTP_FORMAT = """
<|im_start|>system
{}
<|im_end|>
<|im_start|>user
{}
<|im_end|>
<|im_start|>assistant
{}
"""

PROMTP_ANS_FORMAT = """
<|im_start|>system
{}
<|im_end|>
<|im_start|>user
{}
<|im_end|>
<|im_start|>assistant
"""


IN_CONTEXT_PROMPT = {
    "2": "Bạn là một chuyên gia về toán học. Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Đáp án sẽ là một trong các lựa chọn A, B, C, D. Hãy suy nghĩ từng bước một."
    ,"1": "Bạn là một chuyên gia về toán học. Trả lời câu hỏi sau bằng cách đưa ra đáp án chính xác nhất. Hãy suy nghĩ từng bước một."
}

DATASETS = [
    "hllj/vi_gsm8k",
    "hllj/vi_grade_school_math_mcq"
] # list of datasets used for training

BIT_QUANTIZATION = "4bit" # model quantization - can change to 8bit

# LORA CONFIG
R=16 # rank of the LoRA matrices
LORA_ALPHA=32 # scaling factor for LoRA
TARGET_MODULES=[
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj"
] # modules to apply LoRA
LORA_DROPOUT=0.05 # dropout rate for LoRA
BIAS="none" # bias type for LoRA
TASK_TYPE="CAUSAL_LM" # generative task

# MODEL GENERATE CONFIG
MAX_NEW_TOKENS=200 # max length of generated tokens
TEMPERATURE=0.7 # controls randomness in generation
TOP_P=0.7 # nucleus sampling parameter
NUM_RETURN_SEQUENCES=1 # number of generated sequences to return

# DIRECTORY

DATA_PATH = "data/"
TRAINING_TOKENIZED_DATASET = "data/training_tokenized_dataset"
TRAINING_DATASET = "data/training_dataset"

if __name__ == "__main__":
    pass
