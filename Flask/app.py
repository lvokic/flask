from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['DOWNLOAD_FOLDER'] = 'uploads'  # 替换为你的上传文件夹的路径
app.config['UPLOAD_FOLDER'] = 'uploads'


# 启用跨域支持，允许所有来源访问您的API
CORS(app, resources={r"/api-flask/*": {"origins": "http://localhost:8080"}}, supports_credentials=True)


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


@app.route('/api-flask/messages', methods=['GET'])
def get_messages():
    messages = ["Message 1", "Message 2", "Message 3"]
    return jsonify(messages=messages)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
