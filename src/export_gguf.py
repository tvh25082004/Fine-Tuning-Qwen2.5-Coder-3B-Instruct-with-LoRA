"""
Export GGUF — Convert LoRA model đã train sang GGUF để dùng với llama.cpp/Ollama
Yêu cầu: llama.cpp đã được build sẵn
"""
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


def merge_and_save(cfg: Config, adapter_path: str = None) -> Path:
    """Merge LoRA adapter vào base model và lưu dạng HuggingFace."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = adapter_path or f"{cfg.training.output_dir}/final"
    merged_path = Path(cfg.training.output_dir) / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)

    logger.info("📦 Load base model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, path)
    model = model.merge_and_unload()

    logger.info(f"💾 Lưu merged model → {merged_path}")
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))

    return merged_path


def convert_to_gguf(
    merged_path: Path,
    llama_cpp_path: str,
    quantization: str = "q4_k_m",
) -> Path:
    """
    Convert HuggingFace model sang GGUF dùng llama.cpp convert script.

    Args:
        merged_path: Đường dẫn đến model đã merge
        llama_cpp_path: Đường dẫn đến thư mục llama.cpp
        quantization: Mức quantize (q4_k_m, q5_k_m, q8_0, f16)
    """
    output_dir = merged_path.parent / "gguf"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1: Convert sang GGUF F16
    gguf_f16 = output_dir / "model-f16.gguf"
    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"

    logger.info(f"🔄 Convert sang GGUF F16...")
    result = subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(merged_path),
            "--outfile", str(gguf_f16),
            "--outtype", "f16",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout)

    # Bước 2: Quantize
    gguf_quantized = output_dir / f"model-{quantization}.gguf"
    quantize_bin = Path(llama_cpp_path) / "llama-quantize"

    logger.info(f"⚡ Quantize sang {quantization.upper()}...")
    subprocess.run(
        [str(quantize_bin), str(gguf_f16), str(gguf_quantized), quantization.upper()],
        check=True,
    )

    logger.info(f"✅ GGUF đã tạo: {gguf_quantized}")
    return gguf_quantized


def export(
    llama_cpp_path: str = "~/llama.cpp",
    quantization: str = "q4_k_m",
):
    """Main export pipeline."""
    cfg = Config()

    # Merge LoRA → HuggingFace format
    merged = merge_and_save(cfg)

    # Convert → GGUF
    try:
        gguf = convert_to_gguf(merged, llama_cpp_path, quantization)
        print(f"\n🎉 Export thành công!\n   GGUF: {gguf}")
        print(f"\n   Dùng với llama.cpp:")
        print(f"   ./llama-cli -m {gguf} -co -cnv -fa -ngl 99 -n 512")
        print(f"\n   Hoặc với Ollama:")
        print(f"   ollama create senior-dev -f Modelfile")
    except FileNotFoundError:
        logger.error(
            "❌ Không tìm thấy llama.cpp. Cài đặt:\n"
            "   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp\n"
            "   cd ~/llama.cpp && mkdir build && cd build\n"
            "   cmake .. && cmake --build . -j$(nproc)"
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export fine-tuned model sang GGUF")
    parser.add_argument("--llama-cpp", default="~/llama.cpp", help="Path đến llama.cpp")
    parser.add_argument("--quant", default="q4_k_m", help="Quantization type")
    args = parser.parse_args()
    export(args.llama_cpp, args.quant)
