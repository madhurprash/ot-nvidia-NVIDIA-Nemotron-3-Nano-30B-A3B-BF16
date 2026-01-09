import os
import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from peft import PeftModel

BASE_MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"
LORA_DIR   = "/home/ubuntu/devstral-ft/benchmark-agentic-SLMs/Devstral-finetuned-versions/devstral-sft-updated-weightsv0.1"
OUT_DIR    = "/home/ubuntu/devstral-ft/benchmark-agentic-SLMs/Devstral-finetuned-versions/merged-devstral-sft"

os.makedirs(OUT_DIR, exist_ok=True)

# Tokenizer
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok.save_pretrained(OUT_DIR)

# Load base in bf16 (merge should happen in bf16)
base = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Wrap with PEFT adapter
peft_model = PeftModel.from_pretrained(base, LORA_DIR)
peft_model = peft_model.to(torch.bfloat16)
print("Wrapped type:", type(peft_model))
print("Has merge_and_unload:", hasattr(peft_model, "merge_and_unload"))

# Merge
merged = peft_model.merge_and_unload()

# Save merged model
merged.save_pretrained(OUT_DIR, safe_serialization=True)
print(f"✅ Merged model saved to: {OUT_DIR}")
