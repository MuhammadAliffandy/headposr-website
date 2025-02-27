import os
import cv2

# Fungsi untuk memeriksa apakah perangkat video terdeteksi
def check_video_device():
    # Cek apakah perangkat video ada di /dev/video*
    video_devices = [f"/dev/video{i}" for i in range(10)]
    for device in video_devices:
        if os.path.exists(device):
            print(f"{device} terdeteksi!")
        else:
            print(f"{device} tidak ditemukan.")

    # Coba akses kamera pertama (/dev/video0)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Kamera terdeteksi dan berhasil dibuka!")
    else:
        print("Tidak dapat mengakses kamera.")

    cap.release()

# Jalankan pemeriksaan
if __name__ == "__main__":
    check_video_device()
