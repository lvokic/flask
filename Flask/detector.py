import torch
import numpy as np
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2
from random import randint


class Detector(object):

    def __int__(self):
        self.img_size = 640
        self.output_size = 480
        self.threshold = 0.4
        self.device = 'cpu'
        self.model = self.model_load(weights="/best.pt", device=self.device)

    # 初始化模型
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        return model

    def detect_img(self, img=""): # img path
        model = self.model
        output_size = self.output_size


