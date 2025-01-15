# Gunakan image dasar yang memiliki Python
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
    linux-headers-amd64 \
    dkms \
    && rm -rf /var/lib/apt/lists/*

# Install v4l2loopback module dan alat pengguna
RUN git clone https://github.com/umlaeute/v4l2loopback.git /tmp/v4l2loopback && \
    cd /tmp/v4l2loopback && \
    make && make install && \
    depmod -a && \
    rm -rf /tmp/v4l2loopback

# Set working directory di dalam container
WORKDIR /app

# Salin requirements.txt ke dalam container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh aplikasi ke dalam container
COPY . .

# Tambahkan skrip untuk memeriksa perangkat video
COPY check_video.py /app/check_video.py

# Tentukan port yang digunakan aplikasi
EXPOSE 5000

# Tentukan perintah untuk menjalankan check_video.py terlebih dahulu, kemudian aplikasi utama
CMD ["sh", "-c", "python check_video.py && gunicorn --bind 0.0.0.0:5000 app:app"]
