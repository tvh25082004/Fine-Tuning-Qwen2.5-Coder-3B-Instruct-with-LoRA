"""
Inference — Test model sau khi fine-tune
"""
import sys
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_finetuned(cfg: Config, adapter_path: str = None):
    """Load base model + LoRA adapter."""
    path = adapter_path or f"{cfg.training.output_dir}/final"

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    # Merge LoRA adapter
    model = PeftModel.from_pretrained(base, path)
    model = model.merge_and_unload()     # Merge vào base weights
    model.eval()
    logger.info(f"✅ Đã load model từ: {path}")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    user_message: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Generate response từ model."""
    cfg = Config()
    messages = [
        {"role": "system", "content": system_prompt or cfg.system_prompt},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Bỏ phần prompt, chỉ giữ generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_test(cfg: Config):
    """Test model tương tác."""
    print("\n" + "=" * 60)
    print("🤖 Test Fine-tuned Qwen2.5-Coder — Senior Dev Mode")
    print("=" * 60)
    print("Gõ 'quit' để thoát\n")

    model, tokenizer = load_finetuned(cfg)

    # Test cases mẫu
    test_prompts = [
        "Tạo API POST /users với FastAPI, dùng service layer và repository pattern",
        "Fix lỗi: KeyError 'user_id' trong hàm get_user()",
        "Viết Dockerfile tối ưu cho FastAPI app",
        "Giải thích cấu trúc project: controllers/, services/, repositories/, models/",
    ]

    print("📝 Chạy test cases mặc định:\n")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Test {i}] {prompt}")
        print("-" * 40)
        response = generate(model, tokenizer, prompt)
        print(response)
        print()

    # Interactive mode
    print("\n💬 Interactive mode:")
    while True:
        user_input = input("\nBạn: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        response = generate(model, tokenizer, user_input)
        print(f"\nModel: {response}")


if __name__ == "__main__":
    cfg = Config()
    interactive_test(cfg)
