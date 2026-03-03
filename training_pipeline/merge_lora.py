import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with the base model."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Path or HF ID of the base model.",
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HF Hub repo ID to push the merged model to (e.g., 'Vladimirlv/ru-promptriever-qwen3-4b').",
    )

    args = parser.parse_args()

    print(f"Loading base model: {args.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    print(f"Loading LoRA adapter from: {args.lora_model_path} and merging weights")
    model = PeftModel.from_pretrained(base_model, args.lora_model_path)
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        print(f"Pushing merged model to Hub: {args.push_to_hub}")
        merged_model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("Merge completed successfully.")


if __name__ == "__main__":
    main()
