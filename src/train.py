"""
Training Script — Qwen2.5-Coder-3B Senior Dev Fine-tune
Dùng QLoRA (4-bit) + PEFT + TRL SFTTrainer

Hardware:
  - GPU CUDA/ROCm:  load_in_4bit=True, bf16=True
  - CPU (64GB RAM): load_in_4bit=True, fp16=False → chậm hơn nhưng ổn định
"""
import logging
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data_loader import build_dataset, inspect_sample

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def detect_hardware() -> dict:
    """Phát hiện hardware và trả về config phù hợp."""
    info = {
        "device": "cpu",
        "bf16": False,
        "fp16": False,
        "use_4bit": True,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        # Ampere+ mới hỗ trợ bfloat16
        cap = torch.cuda.get_device_capability()
        info["bf16"] = cap[0] >= 8
        info["fp16"] = not info["bf16"]
        logger.info(f"✅ CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️  Không phát hiện CUDA. Dùng CPU — sẽ chậm hơn.")
        logger.info("   (AMD RX 580 không hỗ trợ ROCm trên WSL cho gfx803)")
        logger.info("   Với 64GB RAM, train 3 epochs ~4-6 giờ.")
    return info


def load_model_and_tokenizer(cfg: Config, hw: dict):
    """Load model với quantization phù hợp hardware."""
    bnb_config = None

    if hw["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if not hw["bf16"] else torch.bfloat16,
            bnb_4bit_use_double_quant=True,    # Quantize quantization constants
        )
        logger.info("📦 Load model 4-bit QLoRA")
    else:
        logger.info("📦 Load model full precision (CPU)")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=True,
    )
    # Qwen2.5 dùng eos_token làm pad
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        quantization_config=bnb_config,
        device_map="auto" if hw["device"] == "cuda" else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float16 if hw["fp16"] else torch.float32,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model, cfg: Config):
    """Áp dụng LoRA adapter lên model."""
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train(cfg: Config):
    """Main training function."""
    hw = detect_hardware()

    # Override config theo hardware
    cfg.training.bf16 = hw["bf16"]
    cfg.training.fp16 = hw["fp16"]
    if hw["device"] == "cpu":
        # CPU: giảm batch để tránh OOM
        cfg.training.per_device_train_batch_size = 1
        cfg.training.gradient_accumulation_steps = 8
        cfg.training.optim = "adamw_torch"     # paged_adamw cần CUDA
        logger.info("📟 CPU mode: batch=1, grad_accum=8, optimizer=adamw_torch")

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg, hw)
    model = apply_lora(model, cfg)

    # Load dataset
    dataset = build_dataset(
        path=cfg.data.dataset_path,
        tokenizer=tokenizer,
        val_size=cfg.data.val_size,
        seed=cfg.data.seed,
        max_seq_length=cfg.model.max_seq_length,
    )
    inspect_sample(dataset)

    # Training Arguments
    output = Path(cfg.training.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        report_to=cfg.training.report_to,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=cfg.training.remove_unused_columns,
        optim=cfg.training.optim,
        max_seq_length=cfg.model.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
    )

    logger.info("🚀 Bắt đầu training...")
    trainer.train()

    # Lưu model
    logger.info(f"💾 Lưu model → {output / 'final'}")
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(str(output / "final"))
    logger.info("✅ Training hoàn tất!")


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
