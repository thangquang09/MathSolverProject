import torch
from config_model import (
    get_tokenizer,
    get_model,
    get_generate_config
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def reference(tokenizer, model, prompt, generation_config, device="cpu"):
    encoding = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    model = get_model(
        tokenizer=tokenizer,
    )
    generation_config = get_generate_config(tokenizer, model)

    with open("test_prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    # print(prompt)
    ans = reference(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        generation_config=generation_config
    )
    print(ans)
