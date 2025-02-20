from datasets import load_dataset, Dataset
from constant import (
    DATASETS, MODEL_NAME, PROMTP_FORMAT, IN_CONTEXT_PROMPT, TRAINING_TOKENIZED_DATASET, TRAINING_DATASET)
from config_model import get_tokenizer
from tqdm import tqdm
import random

# ds1_format: 
# {
#     "index": 0,
#     "question": "Natalia đã bán kẹp tóc cho 48 người bạn của cô ấy vào tháng 4, và sau đó cô ấy đã bán nửa số lượng kẹp tóc đó vào tháng 5. Natalia đã bán tổng cộng bao nhiêu kẹp tóc trong tháng 4 và tháng 5?",
#     "explanation": "Natalia đã bán 24 kẹp trong tháng 5.\nNatalia đã bán tổng cộng 72 kẹp trong tháng 4 và tháng 5.",
#     "answer": "72"
# }

# ds2_format: note that "problems"
# {
#   "id": "f9decb7530da8097ebca80315928825e",
#   "question": "Câu 2: Trang 21 - sgk toán lớp 5\nMột gia đình gồm 3 người (bố, mẹ và một con). Bình quân thu nhập hàng tháng 800 000 đồng mỗi người. Nếu gia đình đó có thêm một con nữa mà tổng thu nhập của gia đình không thay đổi thì bình quân thu nhập hàng tháng của mỗi người giảm đi bao nhiêu tiền?",
#   "explanation": "Tổng thu hập bình quân một tháng của gia đình đó là:\n800000 x 3 = 2400000 ( đồng)\nSau khi thêm một người, thu nhập trung bình của một người trong gia đình là:\n2400000 : 4 = 600000 ( đồng)\nVậy so với trước đó, thu nhập bình quân mỗi tháng của một người đã giảm đi:\n800000 - 600000 = 200000 ( đồng)\nĐáp án: 200000 đồng.",
#   "choices": [
#       "A. 180000 đồng.",
#       "B. 250000 đồng.",
#       "C. 220000 đồng.",
#       "D. 200000 đồng."
#   ],
#   "answer": "D. 200000 đồng."
# }

def generate_prompt_tokenize(num_ds: int, tokenizer, question: str, explanation: str, choices: str = None, return_tokenized_sample:bool = False):
    in_context_prompt = IN_CONTEXT_PROMPT[str(num_ds)]
    if num_ds == 1:
        instruction_prompt = f"""
        ### Câu hỏi:
        {question}
        """.strip()
    else:
        instruction_prompt = f"""
        ### Câu hỏi:
        {question}
        ### Các lựa chọn:
        {choices}
        """.strip()
    full_prompt = PROMTP_FORMAT.format(
        in_context_prompt,
        instruction_prompt,
        explanation
    )
    if return_tokenized_sample:
        full_prompt = tokenizer(
            full_prompt,
            padding=True,
            truncation=True,
            max_length=512
        )
    else:
        full_prompt = {
            "prompt": full_prompt
        }
    return full_prompt

def prepare_ds1(tokenizer, training_samples:list, return_tokenized_sample:bool = False):
    print("Preparing Dataset 1...")
    ds1 = load_dataset(DATASETS[0]) # has train and test
    for sample in tqdm(ds1["train"]):
        question = sample["question"]
        explanation = sample["explanation"]

        if explanation == '' or question == '':
            continue
        training_sample = generate_prompt_tokenize(1, tokenizer, question, explanation, return_tokenized_sample=return_tokenized_sample)
        training_samples.append(training_sample)

def prepare_ds2(tokenizer, training_samples:list, return_tokenized_sample:bool = False):
    print("Preparing Dataset 2...")
    ds2 = load_dataset(DATASETS[1]) # has only train
    for sample in tqdm(ds2["train"]):
        for quest in sample['problems']:
            choices = quest["choices"]
            question = quest["question"]
            explanation = quest["explanation"]
            
            if explanation == '' or question == '' or choices == []:
                continue
            try:
                question = question.split("\n \n")[1].strip()
            except:
                continue

            choices = "\n".join(choices)
            training_sample = generate_prompt_tokenize(2, tokenizer, question, explanation, choices, return_tokenized_sample=return_tokenized_sample)
            training_samples.append(training_sample)
    
def prepare_dataset(return_tokenized_sample=True, is_save=False):
    print("Get tokenizer...")
    tokenizer = get_tokenizer()
    training_samples = []
    prepare_ds1(tokenizer, training_samples, return_tokenized_sample)
    prepare_ds2(tokenizer, training_samples, return_tokenized_sample)

    # Shuffling training samples
    random.shuffle(training_samples)

    # Save dataset
    training_dataset = Dataset.from_list(training_samples)
    if is_save:
        if return_tokenized_sample:
            training_dataset.save_to_disk(TRAINING_TOKENIZED_DATASET) # tokenized
        else:
            training_dataset.save_to_disk(TRAINING_DATASET) # prompt
    
    return training_dataset

if __name__ == "__main__":
    choose = input("Tokenize? [Y]/[N]/[B] with [B] is both of datasets: ")
    is_save = input("Save? [Y]/[N]: ")
    is_save = True if is_save.lower() == "y" else False

    if choose.lower() == "y":
        prepare_dataset(True, is_save)
    elif choose.lower() == "b":
        prepare_dataset(False, is_save)
        prepare_dataset(True, is_save)
    else:
        prepare_dataset(False, is_save)
    
