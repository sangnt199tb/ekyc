# Sử dụng Python 3.11 bản gọn nhẹ làm nền tảng
FROM python:3.11-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy danh sách thư viện và tiến hành cài đặt trước (để tận dụng cơ chế cache của Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps vietocr==0.3.13

# Tải sẵn model OCR khi build image để request đầu tiên không phải chờ tải
RUN mkdir -p models \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://vocr.vn/data/vietocr/vgg_transformer.pth', 'models/vgg_transformer.pth')" \
    && python -c "from rapidocr import RapidOCR; RapidOCR()"

# Copy toàn bộ mã nguồn hiện tại của bạn vào container
COPY . .

# Mở cổng 8000 cho API
EXPOSE 8000

# Lệnh chạy ứng dụng khi container khởi động
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]