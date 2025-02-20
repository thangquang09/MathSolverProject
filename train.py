import os
from datasets import load_from_disk
from config_model import get_tokenizer, get_model, bnb_config, lora_config
from constant import TRAINING_DATASET, TRAINING_TOKENIZED_DATASET
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from prepare_datasets import prepare_dataset
from peft import merge_and_unload

if not os.path.exists(TRAINING_TOKENIZED_DATASET):
    print(f"Training dataset not found at {TRAINING_DATASET}")
    training_dataset = prepare_dataset(True, True)
else:
    training_dataset = load_from_disk(TRAINING_TOKENIZED_DATASET)
print("Training Dataset is Loaded")

tokenizer = get_tokenizer()
model = get_model(tokenizer, bnb_config=bnb_config, return_base_model=False, lora_config=lora_config)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir="experiments",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05
)

trainer = Trainer(
    model=model,
    train_dataset=training_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False
trainer.train()

# Merge LoRA parameters into the base model
model = merge_and_unload(model)

# Save the final model
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")