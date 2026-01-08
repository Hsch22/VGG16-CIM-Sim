"""
数据准备与基准模型测试脚本。

本脚本主要完成以下任务：
1. 加载预训练的 VGG16 模型并加载本地权重。
2. 定义标准的图像预处理流程。
3. 读取测试图片并生成基准推理结果。
4. 保存输入张量和模型输出，供后续 CIM 仿真验证使用。
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 设置计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载预训练的 VGG16 模型结构
model = models.vgg16()  # 不传递 weights 参数，手动加载权重

weight_path = ".\\model\\vgg16-397923af.pth"

# 加载本地权重文件
state_dict = torch.load(weight_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

# 设置为评估模式
model.eval()
model.to(device)

print("模型加载完成。")

# 打印模型结构
print(model)

# 打印第一层卷积层的权重形状信息
first_conv_weight = model.features[0].weight.data
print(f"\n--- 第一层卷积 (features.0) 信息 ---")
print(f"权重形状 (Out, In, k, k): {first_conv_weight.shape}")
print(f"偏置形状: {model.features[0].bias.data.shape}")

# VGG16 的 features 部分由一系列层组成：
# features[0]: Conv2d
# features[1]: ReLU
# features[2]: Conv2d
# ...

# 定义预处理流程 (VGG16 的标准预处理)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),  # VGG16 输入尺寸要求为 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 加载并处理测试图片
try:
    # 确保当前目录下存在名为 test_image.jpg 的图片
    img = Image.open("test_image.jpg")

    # 如果图像包含透明通道 (RGBA/LA) 或调色板模式 (P)，转换为 RGB
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    input_tensor = preprocess(img)
    # 增加 Batch 维度 -> [1, 3, 224, 224]
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 运行官方模型，获取基准输出
    with torch.no_grad():
        output_baseline = model(input_batch)

    print(f"\n官方模型推理成功。")
    print(f"输出形状: {output_baseline.shape}")  # 预期形状: [1, 1000]

    # 打印前 5 个预测类别的索引，用于后续验证
    probabilities = torch.nn.functional.softmax(output_baseline[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("Top 5 预测类别索引:", top5_catid.cpu().numpy())

    # 保存输入张量和输出结果到文件，供后续开发步骤使用
    torch.save({'input': input_batch, 'output': output_baseline}, 'baseline_data.pth')
    print("已保存基准数据到 baseline_data.pth")

except FileNotFoundError:
    print("错误：请在当前目录下放一张名为 test_image.jpg 的图片用于测试。")
