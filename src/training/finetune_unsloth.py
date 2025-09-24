import os, time
from datasets import load_dataset
from unsloth import FastLanguageModel

OUTPUT_DIR = "/workspace/adapters/myrun-2025-09-24"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Example tiny dataset; replace with your own in /workspace/data/processed/*.jsonl
# Expect "instruction"/"input"/"output" style or adjust the mapping below.
ds = load_dataset("json", data_files="/workspace/data/processed/train.jsonl")

model_name = "gpt-oss:latest"  # placeholder; use a HF base like "mistralai/Mistral-7B-v0.1"
base = "mistralai/Mistral-7B-v0.1"  # <- real HF model for training

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base,
    load_in_4bit=True,
    max_seq_length=4096,
)

# Simple formatting; adapt to your schema
def format_example(ex):
    instr = ex.get("instruction","")
    inp = ex.get("input","")
    out = ex.get("output","")
    prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return {"input_text": prompt, "labels": out}

ds = ds.map(format_example)

FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"]
)

model = FastLanguageModel.for_training(model)
trainer = FastLanguageModel.get_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    save_steps=1000,
    output_dir=OUTPUT_DIR,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to:", OUTPUT_DIR)