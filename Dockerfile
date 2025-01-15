# Gunakan base image dengan kernel headers yang sesuai
FROM debian:bullseye

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
    && rm -rf /var/lib/apt/lists/*

# Install v4l2loopback module dan tambahkan logging
RUN git clone https://github.com/umlaeute/v4l2loopback.git /tmp/v4l2loopback && \
    cd /tmp/v4l2loopback && \
    make > /tmp/v4l2loopback_build.log 2>&1 && \
    make install && \
    depmod -a && \
    rm -rf /tmp/v4l2loopback

# Set working directory di dalam container
WORKDIR /app

# Salin aplikasi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Tentukan port aplikasi
EXPOSE 5000

# Jalankan aplikasi
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
