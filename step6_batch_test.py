"""
本模块扫描指定文件夹下的所有图片，自动执行预处理、CIM 仿真推理，并统计性能指标。
包含的性能指标：
- 预测结果 (Top-1 Class & Probability)
- 硬件时延 (Latency Cycles)
- 硬件能耗 (Energy uJ, 包含阵列计算和互连开销)
- 实际运行时间 (Wall Clock Time)
"""

import os
import sys
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

try:
    import step4_full_chip_sim as sim
    from step4_full_chip_sim import VGG16_Sim, CONFIG, STATS
except ImportError:
    print("❌ 错误: 找不到 step4_full_chip_sim.py。")
    sys.exit(1)


# 配置与环境准备
EXPERIMENT_DIR = "experiment"
POSSIBLE_MODEL_PATHS = [
    os.path.join("model", "vgg16-397923af.pth"),
    "vgg16-397923af.pth",
    "../model/vgg16-397923af.pth",
]

preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model_path():
    """查找本地权重文件。"""
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_model():
    """
    加载并初始化 CIM 仿真模型。

    Returns:
        VGG16_Sim: 初始化完成的仿真模型。
    """
    model_path = get_model_path()

    if model_path is None:
        print("[System] 本地未找到权重文件，尝试下载 ImageNet 预训练权重...")
        try:
            std_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except:
            std_vgg = models.vgg16(pretrained=True)
    else:
        print(f"[System] 正在加载本地权重: {model_path} ...")
        std_vgg = models.vgg16()
        std_state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
        std_vgg.load_state_dict(std_state_dict)

    # 转换为 CIM 仿真模型
    sim_model = VGG16_Sim(std_vgg)
    sim_model.eval()

    # 配置参数 (与 Step 5 实验设置保持一致)
    CONFIG.cim_array_height = 128
    CONFIG.cim_array_width = 128
    CONFIG.weight_bits = 8
    CONFIG.activation_bits = 8

    print(f"[System] 模型加载完成。")
    print(
        f"    - Array Configuration: {CONFIG.cim_array_height}x{CONFIG.cim_array_width}"
    )
    print(f"    - Precision: Int{CONFIG.weight_bits}")
    print(f"    - Dynamic MAC Energy: {CONFIG.dynamic_mac_energy:.4f} pJ")

    return sim_model


def process_image(image_path):
    """
    读取并预处理单张图片。

    Args:
        image_path (str): 图片文件路径。

    Returns:
        Tensor or None: 预处理后的张量 [1, 3, 224, 224]，若失败返回 None。
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # [1, 3, 224, 224]
        return input_batch
    except Exception as e:
        print(f"[Warning] 无法读取图片 {image_path}: {e}")
        return None


def run_inference(model, input_tensor):
    """
    运行单次推理并收集统计信息。

    Args:
        model (VGG16_Sim): 仿真模型。
        input_tensor (Tensor): 输入数据。

    Returns:
        dict: 包含预测结果和硬件统计信息的字典。
    """
    STATS.reset()

    # 运行推理
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)

        if output.dim() == 3 and output.shape[1] == 1:
            output = output.squeeze(1)

        wall_clock_time = time.time() - start_time

    # 获取预测结果
    probs = torch.softmax(output[0], dim=0)
    top5_prob, top5_ids = torch.topk(probs, 5)

    # 收集硬件统计
    total_energy_pj = 0
    total_latency_cycles = 0
    total_macs = 0

    # 获取当前配置下的动态 MAC 能耗
    current_mac_pj = CONFIG.dynamic_mac_energy

    for layer_name, data in STATS.layers.items():
        # A. MAC 能耗
        e_mac = data["macs"] * current_mac_pj

        # B. 累加能耗
        accum_ops = data.get("accum_ops", 0)
        e_accum = accum_ops * CONFIG.energy_accum_pj

        # C. 接口能耗
        e_adc = data["adc_ops"] * CONFIG.energy_adc_pj
        e_dig = data["digital_ops"] * CONFIG.energy_digital_pj

        total_energy_pj += e_mac + e_accum + e_adc + e_dig
        total_latency_cycles += data["latency_cycles"]
        total_macs += data["macs"]

    return {
        "top5_ids": top5_ids.tolist(),
        "top1_prob": top5_prob[0].item(),
        "energy_uj": total_energy_pj / 1e6,  # pJ -> uJ
        "latency_cycles": total_latency_cycles,
        "macs_ops": total_macs,
        "wall_time": wall_clock_time,
    }


def main():
    # 检查目录
    if not os.path.exists(EXPERIMENT_DIR):
        os.makedirs(EXPERIMENT_DIR)
        print(f"[Info] 创建文件夹: {EXPERIMENT_DIR}/")
        print(f"[Info] 请将测试图片放入该文件夹。")

        print("提示: 你可以找一张 'dog.jpg' 放入 experiment 文件夹。")
        return

    # 扫描图片
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        f for f in os.listdir(EXPERIMENT_DIR) if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"[Warning] {EXPERIMENT_DIR}/ 文件夹下为空。请放入图片后重试。")
        return

    print(f"[System] 找到 {len(image_files)} 张图片，开始批处理...")

    # 加载模型
    model = load_model()

    # 执行批量测试
    print("\n" + "=" * 95)
    print(
        f"{'Image Name':<20} | {'Top-1 ID':<10} | {'Prob':<8} | {'Latency (Cyc)':<15} | {'Energy (uJ)':<12} | {'Time(s)':<8}"
    )
    print("-" * 95)

    results = []

    for img_file in image_files:
        img_path = os.path.join(EXPERIMENT_DIR, img_file)

        # 处理图片
        input_tensor = process_image(img_path)
        if input_tensor is None:
            continue

        # 推理
        stats = run_inference(model, input_tensor)

        # 输出行
        print(
            f"{img_file[:20]:<20} | {stats['top5_ids'][0]:<10} | {stats['top1_prob']:.4f}   | {stats['latency_cycles']:<15} | {stats['energy_uj']:<12.4f} | {stats['wall_time']:.4f}"
        )

        results.append({"file": img_file, **stats})

    print("=" * 95)

    # 统计汇总
    if results:
        avg_latency = sum(r['latency_cycles'] for r in results) / len(results)
        avg_energy = sum(r['energy_uj'] for r in results) / len(results)
        avg_time = sum(r['wall_time'] for r in results) / len(results)

        print(f"\n[Summary Report]")
        print(f" - Total Images  : {len(results)}")
        print(f" - Avg Latency   : {avg_latency:.0f} Cycles")
        print(f" - Avg Energy    : {avg_energy:.4f} uJ")
        print(f" - Avg Wall Time : {avg_time:.4f} s/img")
        print(f"\n[Output] 批量测试完成。")
    else:
        print("\n[Output] 未处理任何有效图片。")


if __name__ == "__main__":
    main()
