import os
import torch
from transformers import 
    (AutoTokenizer, 
     Mistral3ForConditionalGeneration)
from peft import PeftModel

BASE_MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"
LORA_DIR   = "/home/ubuntu/finetuning/benchmark-agentic-SLMs/SFT/trl/outputs/devstral-sft"
OUT_DIR    = "/home/ubuntu/finetuning/benchmark-agentic-SLMs/SFT/trl/outputs/merged-devstral-sft"

os.makedirs(OUT_DIR, exist_ok=True)

# Tokenizer
print(f"Going to load the tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
print(f"Saving tokenizer to {OUT_DIR}...")
tok.save_pretrained(OUT_DIR)

# Load base in bf16 (merge should happen in bf16)
print(f"Going to load the base model in bf16...")
base = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print("Base model loaded.")

# Wrap with PEFT adapter
print(f"Wrapping the base model with PEFT LoRA from {LORA_DIR}...")
peft_model = PeftModel.from_pretrained(base, LORA_DIR)
# Move to bf16 for merging
print("Moving PEFT model to bf16 for merging...")
peft_model = peft_model.to(torch.bfloat16)
print("Wrapped type:", type(peft_model))
print("Has merge_and_unload:", hasattr(peft_model, "merge_and_unload"))

# Merge
print("Merging LoRA weights into the base model...")
merged = peft_model.merge_and_unload()

# Save merged model
merged.save_pretrained(OUT_DIR, safe_serialization=True)
print(f"✅ Merged model saved to: {OUT_DIR}")
