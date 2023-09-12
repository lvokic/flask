import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from models.common import DetectMultiBackend
from utils.torch_utils import select_device


# 1. 定义模型加载函数
def load_model(weights, device='cpu', half=False, dnn=False):
    device = select_device(device)
    half &= device.type != 'cpu'  # 只在CUDA设备上支持半精度
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    # 设置模型的半精度
    if model.pt:
        model.model.half() if half else model.model.float()
    print("模型加载完成!")
    return model


# 2. 加载已训练好的模型
model = load_model(weights="best.pt")  # 替换为您的模型文件的路径

# 3. 准备输入图像
image_path = 'test.jpg'  # 替换为您要检测的图像文件的路径
image = Image.open(image_path)

# 4. 图像预处理
transform = T.Compose([T.ToTensor()])
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度

# 5. 进行推理
with torch.no_grad():
    output = model(input_tensor)

# 6. 处理检测结果
# 假设输出是包含边界框坐标和类别的列表
# 根据您的模型输出结构进行调整
boxes = output[0]['boxes'].cpu().numpy()
scores = output[0]['scores'].cpu().numpy()

# 可视化检测结果
draw = ImageDraw.Draw(image)
for box, score in zip(boxes, scores):
    if score > 0.5:  # 可以根据置信度阈值过滤检测结果
        box = list(map(int, box))
        draw.rectangle(box, outline='red', width=3)

# 7. 保存可视化结果
output_image_path = 'output_image.jpg'  # 替换为保存可视化结果的路径
image.save(output_image_path)

# 8. 显示可视化结果
image.show()
