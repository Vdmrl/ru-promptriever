"""
Main training script for Promptriever (Russian).

Single GPU:
    python train.py --config configs/test_v100.yaml

Multi-GPU with DeepSpeed:
    deepspeed --num_gpus=2 train.py --config configs/raw_dataset_2_5090.yaml
"""

import os
import argparse

import yaml
import torch
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.data import RetrieverDataset, RetrieverCollator
from utils.trainer import RetrieverTrainer


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_wandb(cfg: dict) -> None:
    """Initialize wandb logging. Falls back to .netrc cache if no explicit key is found."""
    wandb_key = cfg.get("wandb_key") or os.environ.get("WANDB_API_KEY")

    if wandb_key:
        wandb.login(key=wandb_key)
        print(
            f"[wandb] Logged in explicitly. Project: {cfg.get('wandb_project', 'ru-promptriever')}"
        )
    else:
        print(
            "[wandb] No explicit API key found in config. Relying on systemic 'wandb login' (.netrc cache)."
        )


def build_model(cfg: dict):
    """Load model in 4-bit quantization (QLoRA) and apply LoRA adapters."""
    model_name = cfg["model_name_or_path"]
    attn_impl = cfg.get("attn_implementation", "sdpa")
    torch_dtype_str = cfg.get("torch_dtype", "float16")

    torch_dtype = getattr(torch, torch_dtype_str, torch.float16)

    use_4bit = cfg.get("load_in_4bit", True)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        quantization_kwargs = {"quantization_config": bnb_config}
    else:
        quantization_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
        attn_implementation=attn_impl,
        dtype=torch_dtype,
        **quantization_kwargs,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get(
            "lora_target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def build_tokenizer(cfg: dict):
    """Load tokenizer and configure pad_token for decoder-only models."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    return tokenizer


def train(cfg: dict) -> None:
    """Main training loop: build model, dataset, trainer, and run."""

    # 1. Logging
    setup_wandb(cfg)

    # 2. Tokenizer
    tokenizer = build_tokenizer(cfg)

    # 3. Model
    model = build_model(cfg)

    # 4. Data
    train_dataset = RetrieverDataset(
        data_path=cfg["train_data_path"],
        num_negatives=cfg.get("num_negatives", 7),
    )

    eval_dataset = None
    eval_path = cfg.get("eval_data_path")
    if eval_path and (eval_path.startswith("hf://") or os.path.exists(eval_path)):
        eval_dataset = RetrieverDataset(
            data_path=eval_path,
            num_negatives=cfg.get("num_negatives", 7),
        )
        print(f"[data] Loaded {len(eval_dataset)} validation examples")

    collator = RetrieverCollator(
        tokenizer=tokenizer,
        max_len_query=cfg.get("max_len_query", 304),
        max_len_passage=cfg.get("max_len_passage", 256),
    )

    print(f"[data] Loaded {len(train_dataset)} training examples")
    print(f"[data] Negatives per query: {cfg.get('num_negatives', 7)}")

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./output_model"),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        learning_rate=cfg.get("learning_rate", 1e-4),
        warmup_steps=cfg.get("warmup_steps", 100),
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        eval_strategy=cfg.get("evaluation_strategy", "no"),
        eval_steps=cfg.get("eval_steps", None),
        logging_steps=cfg.get("logging_steps", 10),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id", None),
        hub_token=os.environ.get("HF_TOKEN", None),  # HF_TOKEN env var is standard
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        deepspeed=cfg.get("deepspeed"),
        remove_unused_columns=False,  # Required for custom collator
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name", cfg.get("wandb_project", "ru-promptriever")),
    )

    # 6. Trainer
    trainer = RetrieverTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        gc_chunk_size=cfg.get("gc_chunk_size", 2),
        temperature=cfg.get("temperature", 0.01),
    )

    # 7. Train
    print("[train] Starting training...")
    trainer.train()

    # 8. Save
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"[train] Model saved to {training_args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Promptriever")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/test_v100.yaml)",
    )
    # Allow DeepSpeed to inject its own CLI arguments
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
