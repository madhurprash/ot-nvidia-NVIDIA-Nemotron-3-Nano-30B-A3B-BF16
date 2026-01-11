"""
Supervised Fine-Tuning (SFT) Script for vLLM-compatible models
using Unsloth for efficient fine-tuning.

Simplified version based on Unsloth notebook - removes unnecessary dtype conversions.
Works with any model supported by Unsloth and vLLM.
"""

import os
import sys
import yaml
import logging
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path="train.yaml"):
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config):
    """Load the model and tokenizer using FastLanguageModel."""
    logger.info("Loading model and tokenizer...")
    model_config = config['model']

    # For multi-GPU training with quantization, we need to ensure each process
    # loads the model to its assigned GPU (based on LOCAL_RANK)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device_map = {"": local_rank}

    logger.info(f"Process local_rank={local_rank}, loading model to device cuda:{local_rank}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config['name'],
        max_seq_length=model_config['max_seq_length'],
        load_in_4bit=model_config['load_in_4bit'],
        load_in_8bit=model_config['load_in_8bit'],
        full_finetuning=model_config['full_finetuning'],
        trust_remote_code=model_config['trust_remote_code'],
        attn_implementation=model_config['attn_implementation'],
        unsloth_force_compile=model_config.get('unsloth_force_compile', False),
        device_map=device_map,  # Ensure each process uses its assigned GPU
    )

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA to the model if enabled."""
    lora_config = config['lora']

    if not lora_config['enabled']:
        logger.info("LoRA is disabled, using full fine-tuning")
        return model

    logger.info("Applying LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
        random_state=lora_config['random_state'],
        use_rslora=lora_config['use_rslora'],
        loftq_config=lora_config.get('loftq_config', None),
    )
    logger.info("LoRA applied successfully")
    return model


def load_and_format_dataset(config, tokenizer):
    """Load and format the dataset for training."""
    logger.info("Loading dataset...")
    dataset_config = config['dataset']

    dataset = load_dataset(
        dataset_config['name'],
        dataset_config['config'],
        split=dataset_config['split']
    )

    # Optionally limit dataset size for debugging
    if dataset_config.get('num_train_samples') is not None:
        num_samples = dataset_config['num_train_samples']
        logger.info(f"Limiting training to {num_samples} samples")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    logger.info(f"Dataset size: {len(dataset)} examples")

    # Format the dataset
    logger.info("Formatting dataset...")

    def formatting_prompts_func(examples):
        """Format conversations using the tokenizer's chat template."""
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Log a sample
    logger.info("Sample formatted text:")
    logger.info("-" * 60)
    logger.info(dataset[0]["text"][:500] + "...")
    logger.info("-" * 60)

    return dataset


def create_trainer(model, tokenizer, dataset, config):
    """Create the SFT trainer."""
    logger.info("Creating trainer...")
    training_config = config['training']

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            max_steps=training_config['max_steps'],
            learning_rate=training_config['learning_rate'],
            logging_steps=training_config['logging_steps'],
            optim=training_config['optim'],
            weight_decay=training_config['weight_decay'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            seed=training_config['seed'],
            output_dir=training_config['output_dir'],
            report_to=training_config['report_to'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            bf16=training_config.get('bf16', False),
            # DDP settings for multi-GPU training
            ddp_find_unused_parameters=training_config.get('ddp_find_unused_parameters', False),
            ddp_backend=training_config.get('ddp_backend', 'nccl'),
        ),
    )
    logger.info("Trainer created successfully")
    return trainer


def train_model(trainer):
    """Train the model."""
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    trainer.train()

    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info("=" * 60)


def save_model(trainer, model, tokenizer, config):
    """Save the trained model."""
    training_config = config['training']
    output_config = config['output']
    output_dir = training_config['output_dir']

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save merged model if enabled
    if output_config['save_merged_model']:
        merged_dir = output_config['merged_output_dir']
        logger.info(f"Saving merged model to {merged_dir}")

        # Merge LoRA weights and save
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )

        logger.info(f"Merged model saved to {merged_dir}")

    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_dir}")
    if output_config['save_merged_model']:
        logger.info(f"Merged model saved to: {merged_dir}")
    logger.info("=" * 60)


def main():
    """Main training function."""
    # Determine config path
    config_path = sys.argv[1] if len(sys.argv) > 1 else "train.yaml"

    # Load configuration
    config = load_config(config_path)

    # Log training configuration
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Max sequence length: {config['model']['max_seq_length']}")
    logger.info(f"Use LoRA: {config['lora']['enabled']}")
    logger.info(f"Output directory: {config['training']['output_dir']}")
    logger.info(f"Batch size per device: {config['training']['per_device_train_batch_size']}")
    logger.info(f"Gradient accumulation steps: {config['training']['gradient_accumulation_steps']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Max steps: {config['training']['max_steps']}")
    logger.info("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA if enabled
    model = apply_lora(model, config)

    # Load and format dataset
    dataset = load_and_format_dataset(config, tokenizer)

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, config)

    # Train model
    train_model(trainer)

    # Save model
    save_model(trainer, model, tokenizer, config)


if __name__ == "__main__":
    main()
