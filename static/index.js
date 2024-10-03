let mediaRecorder;
let audioStream; // Biến toàn cục để lưu trữ stream
const resultLabel = document.getElementById('result');
const startRecording = () => {
    document.getElementById('result-text').textContent = '';
    resultLabel.style.display = 'none';
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        audioStream = stream; // Lưu stream vào biến toàn cục
        mediaRecorder = new MediaRecorder(stream);
        const audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudioToServer(audioBlob);
            audioStream.getTracks().forEach(track => track.stop()); // Dừng mic

            // Gỡ bỏ lớp ghi âm và thay đổi icon về mic
            document.getElementById('app-container').classList.remove('recording');
            document.getElementById('mic-icon').classList.remove('fa-microphone-slash');
            document.getElementById('mic-icon').classList.add('fa-microphone');
            document.getElementById('popup').style.display = 'block';
        };

        mediaRecorder.start();

        // Thêm lớp ghi âm và thay đổi icon
        document.getElementById('app-container').classList.add('recording');
        document.getElementById('mic-icon').classList.remove('fa-microphone');
        document.getElementById('mic-icon').classList.add('fa-microphone-slash');

        // Ghi âm trong 3 giây
        setTimeout(() => {
            mediaRecorder.stop();
        }, 3000);
    }).catch(error => {
        console.error('Error accessing audio devices:', error);
    });
};

const sendAudioToServer = (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        resultLabel.style.display = 'block';
        document.getElementById('result-text').textContent = data.result;
        document.getElementById('popup').style.display = 'none';
    })
    .catch(error => {
        console.error('Error sending audio to server:', error);
        document.getElementById('popup').style.display = 'none';
        resultLabel.style.display = 'block';
        document.getElementById('result-text').textContent = "Unkown";
    });
};

const handleFileUpload = () => {
    const fileInput = document.getElementById('file-input');
    document.getElementById('result-text').textContent = '';
    document.getElementById('popup').style.display = 'block';
    resultLabel.style.display = 'none';
    const file = fileInput.files[0];
    if (file) {
        sendAudioToServer(file);
    } else {
        console.error('No file selected');
    }
};

document.getElementById('start-recording').addEventListener('click', startRecording);
document.getElementById('upload-file').addEventListener('click', handleFileUpload);
document.getElementById('file-input').addEventListener('change', function() {
    var fileName = this.files[0] ? this.files[0].name : 'No file chosen';
    document.getElementById('file-name').textContent = fileName;
});