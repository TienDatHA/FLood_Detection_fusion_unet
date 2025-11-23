# 🐳 Docker Setup cho Hệ thống Phát hiện Lũ lụt

Tài liệu này hướng dẫn cài đặt và sử dụng môi trường Docker hoàn chỉnh cho hệ thống phát hiện lũ lụt.

## 📋 Yêu cầu hệ thống

### 🖥️ Cấu hình tối thiểu
- **OS**: Ubuntu 18.04+ / CentOS 7+ / Windows 10 với WSL2
- **RAM**: 16GB (khuyến nghị 32GB+)
- **Storage**: 50GB trống tối thiểu
- **GPU**: NVIDIA GPU với CUDA support (khuyến nghị RTX 3070+)

### 🛠️ Phần mềm cần thiết
- **Docker Engine**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: nvidia-docker2 (cho GPU support)

## 🔧 Cài đặt Docker

### Ubuntu/Debian
```bash
# Cài đặt Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Thêm user vào docker group
sudo usermod -aG docker $USER

# Cài đặt Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Cài đặt NVIDIA Docker (cho GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### CentOS/RHEL
```bash
# Cài đặt Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io

# Cài đặt NVIDIA Docker
curl -s -L https://nvidia.github.io/nvidia-docker/centos7/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-docker2
sudo systemctl restart docker
```

## 🚀 Khởi động nhanh

### 1️⃣ Chuẩn bị dữ liệu
```bash
# Tạo thư mục data và copy dữ liệu của bạn
mkdir -p ./data
# Copy Sen1Flood11 dataset vào ./data/
# Copy Test_data, Bench_Mark vào ./data/
```

### 2️⃣ Cấu hình môi trường
```bash
# Copy file cấu hình mẫu
cp docker.env docker.env.local

# Chỉnh sửa cấu hình (tuỳ chọn)
nano docker.env.local
```

### 3️⃣ Build và chạy
```bash
# Build image
docker-compose build

# Chạy môi trường chính
docker-compose up -d flood-detection

# Kiểm tra trạng thái
docker-compose ps
```

## 🔄 Các chế độ chạy

### 🖥️ Chế độ Development (Jupyter + TensorBoard)
```bash
# Chạy môi trường development
docker-compose --profile dev up -d

# Truy cập Jupyter Lab
# http://localhost:8889

# Truy cập TensorBoard  
# http://localhost:6007
```

### 🎯 Chế độ Training
```bash
# Chạy training
docker-compose --profile training up

# Theo dõi logs
docker-compose logs -f training
```

### 🔍 Chế độ Inference
```bash
# Chạy inference
docker-compose --profile inference up

# Chạy inference với tham số tùy chỉnh
docker-compose run --rm inference python inference_all.py --region BinhDinh_20171110
```

### 📊 Monitoring với TensorBoard
```bash
# Chạy TensorBoard standalone
docker-compose --profile monitoring up -d tensorboard

# Truy cập: http://localhost:6008
```

## 📁 Cấu trúc Volume mounts

```
./data/              -> /app/data (read-only)
./outputs/           -> /app/outputs
./models/            -> /app/models  
./logs/              -> /app/logs
./training_logs/     -> /app/training_logs
./evaluation_logs/   -> /app/evaluation_logs
./inference_results/ -> /app/inference_results
```

## ⚙️ Cấu hình chi tiết

### 🔧 File docker.env
```bash
# Cấu hình chính
DATA_ROOT=/app/data
REGION_NAME=BinhDinh_20171110
BATCH_SIZE=8
CUDA_VISIBLE_DEVICES=0

# Training parameters
IMG_HEIGHT=256
IMG_WIDTH=256
EPOCHS=50
```

### 🐳 Docker Compose profiles
- **default**: Chế độ cơ bản
- **dev**: Development với Jupyter
- **training**: Chạy training model
- **inference**: Chạy inference
- **monitoring**: TensorBoard monitoring

## 📝 Các lệnh thường dùng

### 🏗️ Build và quản lý images
```bash
# Build tất cả images
docker-compose build

# Build chỉ một service cụ thể
docker-compose build flood-detection

# Xem images đã build
docker images | grep flood-detection

# Xóa images cũ
docker image prune -f
```

### 🔄 Quản lý containers
```bash
# Xem trạng thái
docker-compose ps

# Chạy container
docker-compose up -d flood-detection

# Dừng container
docker-compose stop flood-detection

# Xóa container
docker-compose down

# Vào shell của container
docker-compose exec flood-detection bash

# Chạy lệnh trong container
docker-compose exec flood-detection python --version
```

### 📋 Logs và debugging
```bash
# Xem logs
docker-compose logs flood-detection

# Theo dõi logs real-time
docker-compose logs -f flood-detection

# Xem resource usage
docker stats

# Kiểm tra GPU trong container
docker-compose exec flood-detection nvidia-smi
```

## 🎯 Sử dụng cụ thể

### 🏃‍♂️ Chạy Training
```bash
# Chuẩn bị dữ liệu
mkdir -p ./data/Sen1Flood11
# Copy dữ liệu training vào ./data/

# Chạy training
docker-compose --profile training up

# Hoặc chạy interactive
docker-compose run --rm training python flood.py
```

### 🔍 Chạy Inference
```bash
# Inference với cấu hình mặc định
docker-compose --profile inference up

# Inference với region cụ thể
docker-compose run --rm inference python inference_all.py --region BinhDinh_20171110

# Batch inference nhiều region
docker-compose run --rm inference python run_benchmarks.py
```

### 🧪 Phát triển và Testing
```bash
# Chạy Jupyter cho development
docker-compose --profile dev up -d jupyter

# Chạy tests
docker-compose run --rm flood-detection python -m pytest

# Chạy script tuỳ chỉnh
docker-compose run --rm flood-detection python your_script.py
```

## 🔧 Troubleshooting

### ❗ GPU không được nhận diện
```bash
# Kiểm tra NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Kiểm tra trong container
docker-compose exec flood-detection nvidia-smi
```

### 💾 Vấn đề memory
```bash
# Tăng shared memory
docker-compose run --shm-size=16g flood-detection python flood.py

# Giảm batch size trong docker.env
BATCH_SIZE=4
```

### 📁 Vấn đề permissions
```bash
# Fix permissions cho output directories
sudo chown -R $USER:$USER ./outputs ./models ./logs
```

### 🔌 Port conflicts
```bash
# Thay đổi ports trong docker-compose.yml
ports:
  - "8890:8888"  # Thay vì 8889:8888
```

## 🎯 Production Deployment

### 🏭 Build production image
```bash
# Build optimized production image
docker-compose -f docker-compose.yml build --target prod flood-detection

# Run production
docker-compose -f docker-compose.prod.yml up -d
```

### 🔄 CI/CD Integration
```bash
# Build cho registry
docker build -t your-registry/flood-detection:latest .

# Push to registry
docker push your-registry/flood-detection:latest
```

## 📚 Tài nguyên bổ sung

- **Docker Documentation**: https://docs.docker.com/
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **TensorFlow Docker**: https://www.tensorflow.org/install/docker
- **Docker Compose**: https://docs.docker.com/compose/

---

## 🎉 Hoàn thành!

Bây giờ bạn đã có môi trường Docker hoàn chỉnh cho hệ thống phát hiện lũ lụt! 

Để bắt đầu:
```bash
docker-compose build
docker-compose up -d flood-detection
docker-compose exec flood-detection python --version
```