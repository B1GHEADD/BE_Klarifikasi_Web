import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Aktifkan CORS untuk semua rute di aplikasi Anda

# --- Konfigurasi ---
IMG_WIDTH, IMG_HEIGHT = 150, 150
# MODIFIKASI INI: SESUAIKAN PATH MODEL ANDA
# Jika model.h5 ada di BE/model/best_animal_classifier_fine_tuned.h5
# PASTIKAN PATH INI ABSOLUT ATAU RELATIF DARI LOKASI SCRIPT app.py BERJALAN
MODEL_PATH = './model/best_animal_classifier_fine_tuned.h5'
# PASTIKAN URUTAN KELAS INI SAMA PERSIS DENGAN SAAT TRAINING MODEL ANDA
CLASS_NAMES = ['cats', 'dogs', 'snakes'] # Nama kelas harus sesuai dengan output model Anda

# Muat model saat aplikasi Flask dimulai (hanya sekali)
model = None # Inisialisasi model ke None
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    print("Pastikan file model ada di lokasi yang benar dan formatnya sesuai.")
    print("Aplikasi mungkin tidak berfungsi penuh tanpa model.")

@app.route('/')
def home():
    return "API Klasifikasi Hewan Berjalan (Menggunakan Flask dan Python)!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dimuat. Periksa log server.'}), 500

    if 'file' not in request.files: # Pastikan nama field adalah 'file'
        return jsonify({'error': 'Tidak ada bagian file dalam permintaan. Pastikan FormData menggunakan nama "file".'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Format file tidak didukung. Harap unggah file gambar.'}), 400

    try:
        # Menggunakan PIL (Pillow) untuk membuka gambar dari stream
        img_pil = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_pil = img_pil.resize((IMG_WIDTH, IMG_HEIGHT))

        # Mengkonversi PIL Image ke NumPy array dan preprocessing
        img_array = image.img_to_array(img_pil)
        img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
        img_array = img_array / 255.0 # Normalisasi ke [0, 1]

        predictions = model.predict(img_array)[0]
        
        # Ambil kelas dengan probabilitas tertinggi
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])

        # Atur threshold untuk "Bukan Ketiganya"
        # MODIFIKASI INI: Sesuaikan threshold jika diperlukan
        if confidence < 0.70: # Contoh: Jika keyakinan di bawah 70%, anggap bukan ketiganya
            predicted_class_name = "unknown" # Atau bisa juga langsung return 'Bukan Ketiganya' dari sini

        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': confidence,
            'probabilities': predictions.tolist()
        })
    except Exception as e:
        # Lebih detail error message
        import traceback
        traceback.print_exc() # Cetak traceback ke konsol server untuk debugging
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    print("\nUntuk menjalankan API ini secara lokal (development):")
    print("Pastikan lingkungan virtual Anda aktif, lalu jalankan:")
    print("flask run --host=0.0.0.0 --port=5000") # Gunakan host 0.0.0.0 agar bisa diakses dari jaringan lokal
    print("\nUntuk deployment produksi (direkomendasikan):")
    print("Pastikan lingkungan virtual Anda aktif, lalu jalankan:")
    print("gunicorn --bind 0.0.0.0:5000 app:app")
    
    # Gunakan debug=False untuk produksi, debug=True hanya untuk pengembangan
    app.run(debug=True, host='0.0.0.0', port=5000) # Pastikan debug True hanya saat pengembangan