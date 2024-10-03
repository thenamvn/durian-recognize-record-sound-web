from flask import Flask, render_template, request, jsonify
import os
import uuid
import librosa
import soundfile as sf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import Counter
# from werkzeug.utils import secure_filename
from flask_executor import Executor
from keras.models import load_model
from pydub import AudioSegment

# Cấu hình thư mục để lưu âm thanh tải lên
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Khởi tạo Flask Executor để xử lý không đồng bộ
executor = Executor(app)

# Load mô hình đã huấn luyện
model = load_model('./model/my_model.h5')

# Route cho trang chủ - Load frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route nhận âm thanh từ frontend
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Tạo một UUID cho mỗi tệp tin để tránh trùng lặp tên tệp
    unique_filename = f"{uuid.uuid4()}"
    file_extension = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + file_extension)
    file.save(file_path)

    # Convert .m4a to .wav if necessary
    if file_extension == '.m4a':
        audio = AudioSegment.from_file(file_path, format='m4a')
        wav_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + '.wav')
        audio.export(wav_path, format='wav')
        file_path = wav_path

    # Sử dụng executor.submit để thực hiện xử lý dự đoán không đồng bộ
    future = executor.submit(handle_prediction, file_path)

    # Đợi kết quả từ model (block cho đến khi có kết quả)
    result = future.result()

    # Trả kết quả về frontend
    return jsonify({'result': str(result)})

def create_segments(audio_path, segment_duration=0.3, peak_height=0.75):
    """Tách các đoạn audio xung quanh các đỉnh tín hiệu."""
    y, sr = librosa.load(audio_path, sr=None)

    # Tìm các đỉnh tín hiệu trong đoạn âm thanh
    peaks, _ = find_peaks(y, height=peak_height)

    # Tính toán số lượng mẫu cho mỗi đoạn tách
    segment_samples = int(segment_duration * sr)
    half_segment_samples = segment_samples // 2

    # Tạo thư mục lưu các đoạn tách
    audio_filename = os.path.basename(audio_path)
    audio_name, _ = os.path.splitext(audio_filename)
    segment_dir = os.path.join(app.config['UPLOAD_FOLDER'], audio_name)
    os.makedirs(segment_dir, exist_ok=True)

    # Tách các đoạn âm thanh xung quanh mỗi đỉnh và lưu lại
    segment_filenames = []
    for idx, peak in enumerate(peaks):
        start = max(0, peak - half_segment_samples)
        end = min(len(y), peak + half_segment_samples)
        segment = y[start:end]

        # Lưu đoạn âm thanh vào file .wav
        segment_filename = f"segment_{idx+1}.wav"
        segment_path = os.path.join(segment_dir, segment_filename)
        sf.write(segment_path, segment, sr)
        segment_filenames.append(segment_path)

        # Vẽ spectrogram và lưu
        plot_spectrogram(segment, sr, segment_dir, f"spectrogram_{idx+1}.png")

    return segment_filenames

def plot_spectrogram(y, sr, output_dir, output_filename):
    """Vẽ và lưu spectrogram cho đoạn âm thanh."""
    stft_values = librosa.stft(y, n_fft=1024, hop_length=512)
    stft_magnitude = np.abs(stft_values)
    log_stft = librosa.amplitude_to_db(stft_magnitude, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_stft, sr=sr, cmap='magma')
    plt.tight_layout()

    # Lưu plot vào file ảnh
    output_file = os.path.join(output_dir, output_filename)
    plt.savefig(output_file)
    plt.close()  # Đóng plot để giải phóng bộ nhớ

def predict_segment_images(segment_dir):
    """Dự đoán lớp cho từng ảnh spectrogram trong thư mục."""
    predictions = []
    for segment_filename in os.listdir(segment_dir):
        segment_path = os.path.join(segment_dir, segment_filename)

        # Chỉ dự đoán cho các ảnh .png
        if not segment_filename.lower().endswith('.png'):
            continue

        # Tải và tiền xử lý ảnh
        img = image.load_img(segment_path, target_size=(224, 224))  # Điều chỉnh target_size theo mô hình của bạn
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

        # Dự đoán lớp
        prediction = model.predict(img_array)
        predicted_class = prediction.argmax(axis=-1)[0]
        predictions.append((segment_filename, predicted_class))
        print(f"Predicted class for {segment_filename}: {predicted_class}")

    return predictions

def handle_prediction(file_path):
    """Hàm xử lý dự đoán âm thanh."""
    try:
        # Tạo các đoạn âm thanh
        segment_filenames = create_segments(file_path)

        # Tạo thư mục chứa spectrogram
        spectrogram_dir = os.path.dirname(segment_filenames[0])

        # Dự đoán cho các ảnh spectrogram
        predictions = predict_segment_images(spectrogram_dir)
        
        # Tìm kết quả dự đoán phổ biến nhất (đa số)
        predicted_classes = [pred for _, pred in predictions]
        most_common_result = Counter(predicted_classes).most_common(1)[0][0]

        # Map the most_common_result to a specific string
        result_mapping = {
            0: "mid ripe",
            1: "ripe",
            2: "unripe"
        }
        result_string = result_mapping.get(most_common_result, "Unknown result")
    except Exception as e:
        print(f"Error during prediction: {e}")
        result_string = "Prediction failed"

    # Xóa file và thư mục tạm sau khi xử lý xong
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    if os.path.exists(spectrogram_dir):
        for file in os.listdir(spectrogram_dir):
            os.remove(os.path.join(spectrogram_dir, file))
        os.rmdir(spectrogram_dir)
        print(f"Deleted directory: {spectrogram_dir}")

    return result_string

if __name__ == '__main__':
    app.run(debug=True, threaded=True)