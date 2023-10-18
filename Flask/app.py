from flask import Flask, Response
import cv2
from flask_cors import CORS

app = Flask(__name__)


CORS(app, resources={r"/api-flask/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)

# 使用摄像头捕获视频
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/api-flask/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
