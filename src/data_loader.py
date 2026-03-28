"""
Data Loader — Load và format dataset cho training Qwen2.5-Coder
"""
import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load file JSONL, bỏ qua dòng lỗi."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Bỏ qua dòng {i+1} do lỗi JSON: {e}")
    logger.info(f"Đã load {len(data)} samples từ {path}")
    return data


def format_chat_to_text(sample: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """
    Convert ChatML messages sang text đã apply chat template.
    Qwen2.5 dùng ChatML format với <|im_start|>/<|im_end|>.
    """
    messages = sample.get("messages", [])
    if not messages:
        return {"text": ""}

    # Dùng chat template của tokenizer
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        logger.warning(f"Lỗi apply chat template: {e}")
        text = ""

    return {"text": text}


def build_dataset(
    path: str,
    tokenizer: PreTrainedTokenizer,
    val_size: float = 0.1,
    seed: int = 42,
    max_seq_length: int = 2048,
) -> DatasetDict:
    """
    Load JSONL → format → split train/val → filter quá dài.

    Returns:
        DatasetDict với keys "train" và "test"
    """
    raw_data = load_jsonl(path)
    dataset = Dataset.from_list(raw_data)

    # Format sang text
    dataset = dataset.map(
        lambda x: format_chat_to_text(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting chat templates",
    )

    # Bỏ samples rỗng hoặc quá dài
    def is_valid(sample: dict) -> bool:
        text = sample.get("text", "")
        if not text:
            return False
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids) <= max_seq_length

    before = len(dataset)
    dataset = dataset.filter(is_valid, desc="Filtering invalid/long samples")
    after = len(dataset)
    if before != after:
        logger.info(f"Đã loại {before - after} samples quá dài (>{max_seq_length} tokens)")

    # Train/val split
    split = dataset.train_test_split(test_size=val_size, seed=seed)
    logger.info(
        f"Dataset: {len(split['train'])} train, {len(split['test'])} val"
    )
    return split


def inspect_sample(dataset: DatasetDict, idx: int = 0) -> None:
    """In ra 1 sample để kiểm tra."""
    sample = dataset["train"][idx]
    print("=" * 60)
    print(f"SAMPLE #{idx}")
    print("=" * 60)
    print(sample["text"][:1000])
    print("..." if len(sample["text"]) > 1000 else "")
    print("=" * 60)
