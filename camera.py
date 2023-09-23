import threading
import cv2
import torch
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from pathlib import Path
import time

class CameraDetection:
    def __init__(self, vid_source=0, output_size=480, device='cpu'):
        self.output_size = output_size
        self.vid_source = vid_source
        self.device = device
        self.stopEvent = threading.Event()
        self.model = self.load_model(weights="runs/train/exp5/weights/best.pt")

    @torch.no_grad()
    def load_model(self, weights="", dnn=False):
        device = select_device(self.device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        model.to(device).eval()
        return model

    def detect_camera(self):
        model = self.model
        imgsz = [640, 640]
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
        cap = cv2.VideoCapture(int(self.vid_source) if isinstance(self.vid_source, int) else self.vid_source)

        while not self.stopEvent.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # 转换颜色空间和调整图像大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 640))
            im0s = frame.copy()

            # 预处理图像
            if len(frame.shape) == 3:
                frame = frame[None]
            elif frame.shape[1] != 3:
                frame = frame[:, :3]  # 仅保留前3个通道

            frame = cv2.resize(frame, tuple(reversed(imgsz)), interpolation=cv2.INTER_LINEAR)
            frame = torch.from_numpy(frame).to(device)
            frame = frame.half() if half else frame.float()
            frame /= 255.0

            # 执行目标检测
            pred = model(frame, augment=augment)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):
                seen = 0
                s = ""
                if det is not None and len(det):
                    # 处理检测结果
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = model.names[c]
                        confidence = conf.item()
                        bbox = xyxy.cpu().numpy()
                        bbox = [float(coord) for coord in bbox]

                        # 在图像上绘制边界框和标签
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色边界框
                        cv2.putText(im0s, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 添加标签

                        seen += 1
                        s += f"{seen}: {label}, "  # 添加到显示字符串

                # 显示图像
                cv2.imshow("Camera Detection", im0s)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.stopEvent.clear()

if __name__ == "__main__":
    # 创建 CameraDetection 实例
    camera_detection = CameraDetection(vid_source=0, output_size=480, device='cpu')

    # 启动摄像头检测
    camera_detection.detect_camera()
