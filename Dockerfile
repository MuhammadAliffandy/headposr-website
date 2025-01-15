FROM python:3.11-slim

# Install system dependencies yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    python3-dev \
    v4l-utils \
    linux-headers-$(uname -r) \
    dkms \
    git \
    make \
    gcc \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory di dalam container
WORKDIR /app

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . .

# Tentukan perintah untuk menjalankan aplikasi
COPY check_video.py /app/check_video.py

# Tentukan port yang digunakan aplikasi
EXPOSE 5000

# Perintah untuk menjalankan script check_video dan aplikasi utama
CMD ["sh", "-c", "python check_video.py && gunicorn --bind 0.0.0.0:5000 app:app"]
