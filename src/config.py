"""
Training Configuration — Qwen2.5-Coder-3B Senior Dev Fine-tune
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = True           # QLoRA — tiết kiệm bộ nhớ
    load_in_8bit: bool = False
    dtype: Optional[str] = None         # None = auto detect


@dataclass
class LoraConfig:
    r: int = 16                         # LoRA rank
    lora_alpha: int = 32                # Scaling factor = r * 2
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Layers áp dụng LoRA cho Qwen2.5
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainingConfig:
    output_dir: str = "./output/qwen-senior-dev"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4   # effective batch = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    fp16: bool = False
    bf16: bool = False                      # Set True nếu có Ampere GPU
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = False    # Không dùng validation khi dataset nhỏ
    report_to: str = "none"                 # "wandb" nếu muốn tracking
    dataloader_num_workers: int = 0         # WSL compatibility
    remove_unused_columns: bool = False
    optim: str = "paged_adamw_8bit"        # Tiết kiệm VRAM với AMD


@dataclass
class DataConfig:
    dataset_path: str = "./dataset/senior_dev_dataset.jsonl"
    val_size: float = 0.1                  # 10% cho validation
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # System prompt mặc định
    system_prompt: str = (
        "Bạn là một senior software engineer người Việt với 10+ năm kinh nghiệm. "
        "Chuyên môn: Backend (Python/FastAPI), DevOps (Docker/K8s/CI-CD), AI/ML Engineering. "
        "Bạn luôn:\n"
        "- Code theo design pattern: service layer, repository pattern, dependency injection\n"
        "- Giải thích súc tích, trọng tâm bằng tiếng Việt\n"
        "- Fix bug bằng cách phân tích root cause trước\n"
        "- Tổ chức code module rõ ràng, separation of concerns\n"
        "- Response API luôn chuẩn JSON: {success, data, message}\n"
        "- Viết README ngắn gọn, có ví dụ thực tế"
    )
