"""
vLLM Deployment Script for NVIDIA Nemotron-3-Nano-30B-A3B-BF16

This script demonstrates how to deploy and use the Nemotron model with vLLM.
Supports single generation, batch generation, and streaming inference.
"""
import os
import logging
from utils import load_config
from vllm import LLM, SamplingParams
from typing import Dict, Any, List, Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# load the configuration data
config_data: Dict[str, Any] = load_config()
    
def initialize_model(config):
    """Initialize the vLLM model with appropriate configuration."""
    try:
        # load the model configuration and the vLLM engine configuration
        model_config = config_data['model_information'].get('model_config')
        vllm_config = config_data['model_information'].get('vllm_engine_config')

        # Set PyTorch CUDA memory allocator configuration if specified
        if "pytorch_cuda_alloc_conf" in vllm_config:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = vllm_config["pytorch_cuda_alloc_conf"]

        print(f"Loading {model_config['model_id']} model with vLLM...")
        print(f"  - Max model length: {vllm_config.get('max_model_len', 'default')}")
        print(f"  - GPU memory utilization: {vllm_config.get('gpu_memory_utilization', 'default')}")

        # initialize the LLM with memory optimization from config
        llm = LLM(
            model=model_config["model_id"],
            trust_remote_code=model_config["trust_remote_code"],
            dtype=model_config["dtype"],
            max_model_len=vllm_config.get("max_model_len", 8192),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.85),
            enforce_eager=vllm_config.get("enforce_eager", False),
            swap_space=vllm_config.get("swap_space", 4)
        )
        print("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    return llm


def single_generation_example(llm, config):
    """Example of single prompt generation."""
    print("\n" + "="*60)
    print("Single Generation Example")
    print("="*60)

    inference_params = get_inference_parameters(config)
    params = SamplingParams(
        temperature=inference_params["temperature"],
        max_tokens=inference_params["max_tokens"]
    )
    prompt = "Give me 3 bullet points about vLLM."

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    outputs = llm.generate([prompt], sampling_params=params)
    response = outputs[0].outputs[0].text

    print(f"\nResponse:\n{response}")


def batch_generation_example(llm, config):
    """Example of batch prompt generation."""
    print("\n" + "="*60)
    print("Batch Generation Example")
    print("="*60)

    inference_params = get_inference_parameters(config)
    params = SamplingParams(
        temperature=inference_params["temperature"],
        max_tokens=inference_params["max_tokens"]
    )
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Explain quantum computing in one sentence:"
    ]
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params=params)
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {output.outputs[0].text}")


def custom_generation(llm, config, prompt, temperature=None, max_tokens=None):
    """Run custom generation with specified parameters."""
    inference_params = get_inference_parameters(config)
    # Use provided values or fall back to config defaults
    temp = temperature if temperature is not None else inference_params["temperature"]
    max_tok = max_tokens if max_tokens is not None else inference_params["max_tokens"]
    params = SamplingParams(temperature=temp, max_tokens=max_tok)
    outputs = llm.generate([prompt], sampling_params=params)
    return outputs[0].outputs[0].text


def main():
    """Main function to run vLLM deployment examples."""
    print("NVIDIA Nemotron-3-Nano-30B vLLM Deployment")
    print("="*60)

    # Load configuration
    config = load_config()

    # Initialize model
    llm = initialize_model(config)

    # Run examples
    single_generation_example(llm, config)
    batch_generation_example(llm, config)

    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Enter your prompts (or 'quit' to exit)")

    while True:
        try:
            user_prompt = input("\nPrompt: ").strip()

            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_prompt:
                continue

            print("\nGenerating response...")
            response = custom_generation(llm, config, user_prompt)
            print(f"\nResponse:\n{response}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
