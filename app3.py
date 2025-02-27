from flask import Flask, render_template, Response
import os
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from headposr_model import HeadPosr  # Ganti dengan definisi model Anda
from scipy.spatial.transform import Rotation as R
import dlib

# Inisialisasi Flask
app = Flask(__name__)

# Inisialisasi detector wajah
detector = dlib.get_frontal_face_detector()

# Load model HeadPosr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./model/biwieh64_new_fold_1.pth"
model = HeadPosr()  # Ganti dengan definisi model Anda
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fungsi untuk mendapatkan deskripsi orientasi kepala
def get_orientation_description(yaw, pitch, roll):
    description = []

    # Deskripsi berdasarkan yaw
    if yaw > 0.2:
        description.append("Menghadap kanan")
    elif yaw < -0.2:
        description.append("Menghadap kiri")
    else:
        description.append("Menghadap depan")

    # Deskripsi berdasarkan pitch
    if pitch > 0.2:
        description.append("Sedikit menunduk")
    elif pitch < -0.2:
        description.append("Sedikit mendongak")
    else:
        description.append("Datar")

    # Deskripsi berdasarkan roll
    if roll > 0.2:
        description.append("Kepala miring ke kanan")
    elif roll < -0.2:
        description.append("Kepala miring ke kiri")
    else:
        description.append("Tidak miring")

    return ', '.join(description)

# Fungsi untuk menggambar Euler angles dan menampilkan teks
def draw_euler_angles(image, yaw, pitch, roll, bbox, size=50):
    x, y, w, h = bbox
    cx, cy = (x + w // 2, y + h // 2)

    # Gunakan SciPy untuk mendapatkan matriks rotasi dari Euler angles
    rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    R_matrix = rotation.as_matrix()

    # Proyeksi sumbu pada gambar
    x1 = size * R_matrix[0, 0] + cx
    y1 = size * R_matrix[1, 0] + cy
    x2 = size * R_matrix[0, 1] + cx
    y2 = size * R_matrix[1, 1] + cy
    x3 = -size * R_matrix[0, 2] + cx
    y3 = -size * R_matrix[1, 2] + cy

    # Gambarkan garis sumbu pada gambar
    cv2.line(image, (int(cx), int(cy)), (int(x1), int(y1)), (0, 0, 255), 2)  # Yaw (merah)
    cv2.line(image, (int(cx), int(cy)), (int(x2), int(y2)), (0, 255, 0), 2)  # Pitch (hijau)
    cv2.line(image, (int(cx), int(cy)), (int(x3), int(y3)), (255, 0, 0), 2)  # Roll (biru)

    return image

# Fungsi generator untuk stream video
def generate_frames():
    cap = cv2.VideoCapture(0)  # Buka kamera (gunakan 0 untuk default webcam)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi wajah menggunakan dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Default deskripsi untuk ditampilkan di pojok kanan atas
        orientation_text = "Tidak ada wajah terdeteksi"

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_region = frame[y:y + h, x:x + w]

            # Preprocess wajah
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            face_rgb_pil = Image.fromarray(face_rgb)
            input_image = transform(face_rgb_pil).unsqueeze(0).to(device)

            # Prediksi dengan model
            with torch.no_grad():
                output = model(input_image)
                yaw, pitch, roll = output[0].cpu().numpy()

            # Gambar Euler angles pada gambar asli
            frame = draw_euler_angles(frame, yaw, pitch, roll, (x, y, w, h))

            # Perbarui deskripsi orientasi untuk ditampilkan
            orientation_text = get_orientation_description(yaw, pitch, roll)

        # Tambahkan teks orientasi di pojok kanan atas
        cv2.putText(frame, orientation_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode frame menjadi format JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Kirim frame ke klien
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Menjalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True)
