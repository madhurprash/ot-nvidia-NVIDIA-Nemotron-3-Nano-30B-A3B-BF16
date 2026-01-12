# train_grpo.py
import os
import sys
import yaml
import torch_dtype
from pathlib import Path
from trl import GRPOTrainer
from datasets import load_dataset
from trl.rewards import accuracy_reward
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

class LossThresholdCallback(TrainerCallback):
    """
    Callback that stops training when the loss falls below a specified threshold.
    """
    def __init__(self, loss_threshold: float = 0.1):
        self.loss_threshold = loss_threshold
        self.best_loss = float('inf')

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """
        Called after logging the last logs.
        """
        if logs is not None and 'loss' in logs:
            current_loss = logs['loss']
            self.best_loss = min(self.best_loss, current_loss)

            print(f"Current loss: {current_loss:.4f} | Best loss: {self.best_loss:.4f} | Threshold: {self.loss_threshold}")

            if current_loss <= self.loss_threshold:
                print(f"\nLoss threshold reached! Current loss ({current_loss:.4f}) <= threshold ({self.loss_threshold})")
                print("Stopping training early...")
                control.should_training_stop = True

        return control

def load_config(config_path="train.yaml"):
    """Load configuration from YAML file."""
    print(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(config):
    """Load the model and tokenizer."""
    print(f"Loading the tokenizer...")
    model_config = config['model']
    model_name = model_config['name']

    # Check if model is already cached
    cache_dir = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface/hub'))
    print(f"Checking cache directory: {cache_dir}")

    # Try to load from cache first (faster for distributed training)
    print(f"Attempting to load tokenizer from cache...")
    # Use AutoTokenizer which will automatically select the correct backend
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config['trust_remote_code']
    )
    print(f"Loaded tokenizer from cache successfully: {tokenizer}")
    print(f"Attempting to load model from cache...")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_config['attn_implementation'],    # IMPORTANT (see below)
    )
    print("Model loaded from cache successfully!")
    return model, tokenizer

def load_and_format_dataset(config, tokenizer):
    """Load and format the dataset for training."""
    print("Loading dataset...")
    dataset_config = config['dataset']
    # Load dataset
    dataset = load_dataset(
        dataset_config['name'],
        dataset_config.get('config'),
        split=dataset_config['split']
    )
    # Optionally limit dataset size for testing
    if dataset_config.get('num_train_samples') is not None:
        num_samples = dataset_config['num_train_samples']
        print(f"Limiting training to {num_samples} samples")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    print(f"Dataset size: {len(dataset)} examples")
    
    # Format dataset using chat template
    def formatting_func(example):
        """Format conversations using the tokenizer's chat template."""
        conversation = example["conversations"]

        # Filter out messages with empty content
        filtered_conversation = [
            msg for msg in conversation
            if msg.get("content", "").strip() != ""
        ]

        # Skip if no valid messages remain
        if not filtered_conversation:
            return {"text": ""}

        # Ensure roles alternate between user and assistant
        # Merge consecutive messages from the same role
        normalized_conversation = []
        for msg in filtered_conversation:
            role = msg.get("role", msg.get("from", "")).lower()
            content = msg.get("content", "")

            # Normalize role names
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant", "bot"]:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                # Skip unknown roles
                continue

            # If the last message has the same role, merge the content
            if normalized_conversation and normalized_conversation[-1]["role"] == role:
                normalized_conversation[-1]["content"] += "\n\n" + content
            else:
                normalized_conversation.append({"role": role, "content": content})

        # Ensure conversation starts with user or system message
        if not normalized_conversation or normalized_conversation[0]["role"] == "assistant":
            return {"text": ""}

        # Ensure conversation ends with assistant message for training
        if normalized_conversation[-1]["role"] != "assistant":
            return {"text": ""}

        try:
            text = tokenizer.apply_chat_template(
                normalized_conversation,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=False,
            )
            return {"text": text}
        except Exception as e:
            # If template application fails, skip this example
            print(f"Warning: Failed to apply chat template: {e}")
            return {"text": ""}
    
    dataset = dataset.map(formatting_func, remove_columns=dataset.column_names)
    # Filter out examples with empty text
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: x["text"].strip() != "")
    filtered_size = len(dataset)
    if filtered_size < original_size:
        print(f"Filtered out {original_size - filtered_size} examples with empty content")
    print(f"Final dataset size: {filtered_size} examples")
    # Log a sample
    print("\nSample formatted text:")
    print("-" * 80)
    print(dataset[0]["text"])
    print("-" * 80)
    return dataset




# trainer = GRPOTrainer(
#     model="Qwen/Qwen2-0.5B-Instruct",
#     reward_funcs=accuracy_reward,
#     train_dataset=dataset,
# )
# trainer.train()