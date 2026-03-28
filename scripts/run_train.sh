#!/bin/bash
# Hướng dẫn chạy train Qwen2.5-Coder-3B và tự động ghi log

# Thư mục gốc project
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# Kiểm tra Virtual Env
if [ ! -d "venv" ]; then
    echo "❌ Lỗi: Không tìm thấy thư mục venv."
    echo "👉 Hãy chạy ./scripts/setup_wsl.sh trước!"
    exit 1
fi

source venv/bin/activate

# Tạo dataset nếu chưa có
if [ ! -f "dataset/senior_dev_dataset.jsonl" ]; then
    echo "🛠️ Tạo dataset mẫu (Senior Dev Data)..."
    python dataset/build_dataset.py
fi

# Thư mục log file
mkdir -p logs
LOG_FILE="logs/fine_tune_$(date +'%Y%m%d_%H%M%S').log"

echo "🚀 Bắt đầu quá trình Fine-tuning..."
echo "📄 Nhật ký (logs) sẽ được viết vừa lên màn hình vừa xuất ra file:"
echo "👉 $LOG_FILE"
echo "------------------------------------------------------------"

# Lệnh `2>&1 | tee` giúp vừa ghi lỗi/cảnh báo, vừa hiện lên màn hình, vừa lưu vào file text
export PYTHONUNBUFFERED=1
python src/train.py 2>&1 | tee "$LOG_FILE"

echo "------------------------------------------------------------"
echo "✅ Quá trình kết thúc (Xem log cụ thể ở file $LOG_FILE)"
