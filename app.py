import os
import cv2
import dlib
import torch
from PIL import Image
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from torchvision.transforms import transforms
from headposr_model import HeadPosr  # Import arsitektur model Anda
from scipy.spatial.transform import Rotation as R

# Inisialisasi Flask
app = Flask(__name__)

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

# Inisialisasi dlib face detector
detector = dlib.get_frontal_face_detector()

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

# Fungsi untuk stream kamera
def generate_frames():
    # cap = cv2.VideoCapture(0)
    # # Mencoba untuk membuka perangkat video /dev/video0, /dev/video1, dst.
    for i in range(10):  # Menguji dari /dev/video0 sampai /dev/video9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Video device /dev/video{i} berhasil dibuka!")
            cap.release()
        else:
            print(f"Video device /dev/video{i} tidak ditemukan atau gagal dibuka.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Deteksi wajah menggunakan dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

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

            # Gambar Euler angles pada frame asli
            frame = draw_euler_angles(frame, yaw, pitch, roll, (x, y, w, h))

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Fungsi untuk memproses gambar yang diunggah
@app.route('/upload', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Proses deteksi wajah
            image = cv2.imread(filepath)
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

            # Simpan hasil gambar
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{file.filename}")
            cv2.imwrite(output_path, image)

            return render_template('index.html', uploaded_image=file.filename, processed_image=f"output_{file.filename}")

    return render_template('index.html')

# Routing untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Routing untuk video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
