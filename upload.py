import cv2
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from test import MainWindow

app = Flask(__name__)

main = MainWindow()
app.config['DOWNLOAD_FOLDER'] = 'Flask/downloads'  # 替换为你的上传文件夹的路径
app.config['UPLOAD_FOLDER'] = 'Flask/uploads'
app.config['RESULTS_FOLDER'] = 'Flask/results'  # 存储检测结果图像的文件夹路径

# 启用跨域支持，允许所有来源访问您的API
CORS(app, resources={r"/api-flask/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)

frame_number = 0
save_folder = 'Flask/captured'


# 摄像头捕获
def camera_generator():
    # 打开摄像头，可以更改摄像头索引或视频文件路径
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 构造图像文件的路径
        image_path = os.path.join(save_folder, f'frame_{frame_number}.jpg')

        # 将帧写入磁盘
        cv2.imwrite(image_path, frame)

        # 执行目标检测
        detections = main.detect_objects(image_path)  # 调用您的目标检测方法

        # 在图像上绘制检测结果
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bbox']

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        # 将图像转换为JPEG格式
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # 将JPEG格式的图像作为生成器的输出
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()


# 添加一个路由处理OPTIONS请求
@app.route('/api-flask/upload/', methods=['OPTIONS'])
def handle_options_request():
    response = jsonify()
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8080'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api-flask/upload', methods=['GET'])
def download_file(filename):
    try:
        return send_file(f"{app.config['DOWNLOAD_FOLDER']}/{filename}", as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404


@app.route('/api-flask/upload/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File successfully uploaded'})


@app.route('/api-flask/detect', methods=['POST'])
def detect_objects():
    try:
        # 获取上传的图像文件
        img_file = request.files['image']

        # 生成唯一的文件名
        img_filename = secure_filename(img_file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

        # 保存图像到服务器
        img_file.save(img_path)

        # 执行目标检测并保存检测结果图像到指定路径
        save_result_path = os.path.join(app.config['RESULTS_FOLDER'], 'result_image.jpg')
        main.image_with_detections(img_path, save_result_path)

        # 返回检测结果图像文件
        return send_file(save_result_path, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)})


# 新增路由用于传输视频流
# @app.route('/api-flask/video_feed', methods=['GET'])
# def video_feed():
#     return Response(camera_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api-flask/video_feed', methods=['GET'])
def video_feed():
    return Response(main.detect_camera_results(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
