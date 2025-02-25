import os
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from config_model import (
    bnb_config,
    get_generate_config,
    get_tokenizer,
    lora_config,
    print_trainable_parameters
)
from constant import MODEL_NAME, TRAINING_DATASET, TRAINING_TOKENIZED_DATASET
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training
)
from prepare_datasets import prepare_dataset

# Load Training Dataset
if not os.path.exists(TRAINING_TOKENIZED_DATASET):
    print(f"Training dataset not found at {TRAINING_DATASET}")
    training_dataset = prepare_dataset(True, True)
else:
    training_dataset = load_from_disk(TRAINING_TOKENIZED_DATASET)
print("Training Dataset is Loaded")

# Get Tokenizer
tokenizer = get_tokenizer()

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("Trainable Params after LoRA config:")
print_trainable_parameters(model)

generation_config = get_generate_config(tokenizer, model)

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir="model/experiments",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    report_to="none",
    ddp_find_unused_parameters=False,  # Hỗ trợ multi-GPU
)

trainer = Trainer(
    model=model,
    train_dataset=training_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

print("Training...")
trainer.train()

# Save the final model and tokenizer
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")