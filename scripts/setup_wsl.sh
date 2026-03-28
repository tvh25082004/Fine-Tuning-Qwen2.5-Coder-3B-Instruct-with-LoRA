#!/bin/bash
# Setup environment cho WSL Ubuntu
# Lưu ý: Script này chạy trực tiếp trên thư mục wsl

echo "🚀 Bắt đầu cài đặt môi trường Fine-tuning Qwen2.5-Coder-3B trên WSL"

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Không tìm thấy Python3. Hãy cài đặt: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "📦 Tạo virtual environment (venv)..."
python3 -m venv venv
source venv/bin/activate

echo "🔄 Cập nhật pip..."
pip install --upgrade pip wheel setuptools

echo "⬇️ Đang cài đặt thư viện từ requirements.txt..."
pip install -r requirements.txt

# Kiểm tra PyTorch
echo "🔍 Kiểm tra cài đặt PyTorch và Hardware (CUDA/ROCm/CPU)..."
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('✅ Tìm thấy CUDA/ROCm (GPU):', torch.cuda.get_device_name(0))
else:
    print('⚠️ Không tìm thấy GPU, hệ thống sẽ sử dụng CPU (Khuyên dùng 64GB RAM).')
"

echo "✅ Cài đặt môi trường hoàn tất."
echo "👉 Để kích hoạt môi trường: source venv/bin/activate"
