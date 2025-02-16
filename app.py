from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import os
import cv2
import torch
from PIL import Image
import numpy as np
from io import BytesIO
from torchvision.transforms import transforms
from headposr_model import HeadPosr  # Ganti dengan definisi model Anda
from scipy.spatial.transform import Rotation as R
import base64
import dlib

# Inisialisasi Flask dan SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Inisialisasi detector wajah
detector = dlib.get_frontal_face_detector()
# Direktori untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

# Fungsi untuk menggambar Euler angles
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

# Fungsi untuk menangani stream video dari klien
@socketio.on('video_stream')
def handle_video_stream(frame_data):
    # Decode base64 menjadi gambar
    image_data = base64.b64decode(frame_data)
    image = Image.open(BytesIO(image_data))

    # Convert gambar menjadi format OpenCV (BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Deteksi wajah menggunakan dlib
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = image[y:y + h, x:x + w]

        # Preprocess wajah
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        face_rgb_pil = Image.fromarray(face_rgb)
        input_image = transform(face_rgb_pil).unsqueeze(0).to(device)

        # Prediksi dengan model
        with torch.no_grad():
            output = model(input_image)
            yaw, pitch, roll = output[0].cpu().numpy()

        # Gambar Euler angles pada gambar asli
        image = draw_euler_angles(image, yaw, pitch, roll, (x, y, w, h))

    # Encode frame menjadi base64 dan kirimkan kembali ke klien
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    emit('video_frame', encoded_image)  # Kirim frame yang diproses kembali ke klien

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

# Menjalankan aplikasi Flask dengan SocketIO
if __name__ == "__main__":
    socketio.run(app, debug=True)

