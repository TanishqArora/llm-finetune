import os
import ollama

MODEL = os.environ.get("MODEL_NAME", "mymodel:finetuned")

resp = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Give me a 1-sentence summary of LoRA fine-tuning."}],
    options={"temperature": 0.2}
)
print(resp["message"]["content"])