import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import os
from cnn import MiniVGG # 确保文件名是小写 cnn.py

# --- 1. 加载模型 ---
MODEL_PATH = 'model/MiniVGG/minivgg_003.pth' # 确保路径和模型版本正确
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = MiniVGG()
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. 预测函数 ---
def predict(input_image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    pil_image = Image.fromarray(input_image)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = net(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
    
    confidences = {classes[i]: float(probabilities[i]) for i in range(10)}
    return confidences

# --- 3. Gradio 界面 ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="上传一张图片来猜猜看！"),
    outputs=gr.Label(num_top_classes=3, label="电脑的猜测结果"),
    title="'猜猜看'小游戏",
    description="这是一个基于CIFAR-10数据集训练的图像分类器。"
)

# --- 4. 启动 ---
iface.launch(share=True)