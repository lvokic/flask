import threading
import os
import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import shutil
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


class MainWindow:
    def __init__(self):
        self.output_size = 100
        self.img2predict = ""
        self.device = 'cpu'
        self.vid_source = '0'
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model = self.model_load(weights="runs/train/exp5/weights/best.pt",
                                     device=self.device)

    @torch.no_grad()
    def model_load(self, weights="", device='', half=False, dnn=False):
        device = select_device(device)
        half &= device.type != 'cpu'
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        half &= pt and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    def detect_objects(self, img_path):
        global dataset
        model = self.model
        output_size = self.output_size
        imgsz = [224, 224]
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        device = self.device
        classes = None
        agnostic_nms = False
        augment = False
        half = False
        dnn = False

        source = img_path
        source = str(source)
        device = select_device(self.device)
        webcam = False
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)
        save_img = False

        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))

        results = []

        if webcam:
            pass
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255

            if len(im.shape) == 3:
                im = im[None]

            pred = model(im, augment=augment)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):
                det_dict = {}
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c]
                    confidence = conf.item()
                    bbox = np.array(xyxy)  # 将xyxy转换为NumPy数组
                    bbox = [float(coord) for coord in bbox]
                    det_dict = {
                        'label': label,
                        'confidence': confidence,
                        'bbox': bbox
                    }
                    results.append(det_dict)

        return results

    def display_image_with_detections(self, img_path, save_result_path):
        # 执行目标检测并获取结果
        detections = self.detect_objects(img_path)

        # 打开图像文件
        img = cv2.imread(img_path)

        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bbox']

            # 提取边界框坐标
            x1, y1, x2, y2 = map(int, bbox)

            # 在图像上绘制边界框和标签
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色边界框
            cv2.putText(img, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)  # 添加标签

        # 保存带有检测结果的新图像
        result_image_path = "result_image.jpg"
        cv2.imwrite(save_result_path, img)

        # 显示结果图像
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_with_detections(self, img_path, save_result_path):
        # 执行目标检测并获取结果
        detections = self.detect_objects(img_path)

        # 打开图像文件
        img = cv2.imread(img_path)

        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bbox']

            # 提取边界框坐标
            x1, y1, x2, y2 = map(int, bbox)

            # 在图像上绘制边界框和标签
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色边界框
            cv2.putText(img, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)  # 添加标签

        # 保存带有检测结果的新图像到指定路径
        cv2.imwrite(save_result_path, img)

        # 返回结果图像的保存路径
        return save_result_path

    def detect_camera(self):
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # 模型期望的输入尺寸
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        device = self.device
        classes = None
        agnostic_nms = False
        augment = False
        half = False
        dnn = False

        # 打开摄像头
        cap = cv2.VideoCapture(int(self.vid_source) if self.webcam else self.vid_source)

        while not self.stopEvent.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 调整图像大小以匹配模型的输入尺寸
            frame = cv2.resize(frame, tuple(reversed(imgsz)), interpolation=cv2.INTER_LINEAR)

            # 2. 将图像从 BGR 格式转换为 RGB 格式
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. 将图像数据从 numpy 数组转换为 PyTorch 张量，并标准化像素值到 [0, 1]
            frame = torch.from_numpy(frame / 255.0).float()

            frame = frame.permute(2, 0, 1)  # 将通道维度调整为 [3, H, W]
            frame = frame.unsqueeze(0)  # 添加批次维度 [1, 3, H, W]

            frame = frame.to(device)

            # 执行目标检测
            pred = model(frame, augment=augment)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            frame_np = frame[0].permute(1, 2, 0).cpu().numpy()  # 转换为NumPy数组，注意通道维度的顺序

            for i, det in enumerate(pred):
                seen = 0
                s = ""
                if det is not None and len(det):
                    # 处理检测结果
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = model.names[c]
                        confidence = conf.item()
                        bbox = np.array(xyxy)  # 将xyxy转换为NumPy数组
                        bbox = [int(coord) for coord in bbox]

                        # 在图像上绘制边界框和标签
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色边界框
                        cv2.putText(frame_np, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 添加标签

                        seen += 1
                        s += f"{seen}: {label}, "  # 添加到显示字符串

            # 显示图像
            cv2.imshow("Camera Detection", frame_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        # cv2.destroyAllWindows()
        self.stopEvent.clear()

    def detect_camera_results(self):
        model = self.model
        output_size = self.output_size
        imgsz = [640, 640]  # 模型期望的输入尺寸
        conf_thres = 0.25
        iou_thres = 0.45
        max_det = 1000
        device = self.device
        classes = None
        agnostic_nms = False
        augment = False
        half = False
        dnn = False

        # 打开摄像头
        cap = cv2.VideoCapture(int(self.vid_source) if self.webcam else self.vid_source)

        while not self.stopEvent.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 调整图像大小以匹配模型的输入尺寸
            frame = cv2.resize(frame, tuple(reversed(imgsz)), interpolation=cv2.INTER_LINEAR)

            # 2. 将图像从 BGR 格式转换为 RGB 格式
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3. 将图像数据从 numpy 数组转换为 PyTorch 张量，并标准化像素值到 [0, 1]
            frame = torch.from_numpy(frame / 255.0).float()

            frame = frame.permute(2, 0, 1)  # 将通道维度调整为 [3, H, W]
            frame = frame.unsqueeze(0)  # 添加批次维度 [1, 3, H, W]

            frame = frame.to(device)

            # 执行目标检测
            pred = model(frame, augment=augment)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            frame_np = frame[0].permute(1, 2, 0).cpu().numpy()  # 转换为NumPy数组，注意通道维度的顺序

            for i, det in enumerate(pred):
                seen = 0
                s = ""
                if det is not None and len(det):
                    # 处理检测结果
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = model.names[c]
                        confidence = conf.item()
                        bbox = np.array(xyxy)  # 将xyxy转换为NumPy数组
                        bbox = [int(coord) for coord in bbox]

                        # 在图像上绘制边界框和标签
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色边界框
                        cv2.putText(frame_np, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 添加标签

                        seen += 1
                        s += f"{seen}: {label}, "  # 添加到显示字符串

             # 编码帧为JPEG格式
            # cv2.imshow("Camera Detection", frame_np)


            cv2.imshow("Camera Detection", frame_np)
            _, jpeg = cv2.imencode('.jpg', frame_np)
            frame_data = jpeg.tobytes()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # yield (b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        cap.release()
        self.stopEvent.clear()

if __name__ == "__main__":
    # 创建 MainWindow 实例
    main_window = MainWindow()

    main_window.detect_camera_results()
    # 准备要检测的图像文件
    image_path = "test.jpg"
    save_result_path = "result_image.jpg"

    # 显示图像和检测结果
    main_window.display_image_with_detections(image_path, save_result_path)
