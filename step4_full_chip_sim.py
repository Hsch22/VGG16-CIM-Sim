"""
Step 4: 芯片级全功能仿真 (Full Chip Simulation)。

本模块引入物理感知的能耗模型和硬件行为模拟，重点关注：
- 时延 (Latency): 基于计算周期数。
- 能耗 (Energy): 包含阵列内计算 (Intra-Array) 和阵列间累加 (Inter-Tile) 的动态权衡。
- 精度 (Accuracy): 引入量化 (Quantization) 模拟。

核心权衡逻辑：
- 阵列越大 -> 单次 MAC 能耗越高 (位线寄生电容增加)。
- 阵列越小 -> 累加能耗越高 (需要更多部分和累加操作)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class SimConfig:
    """
    仿真配置参数类。

    管理架构参数、精度设置和物理能耗模型参数。
    """

    # --- 架构参数 ---
    cim_array_height: int = 128  # Word Lines (决定位线长度/寄生电容)
    cim_array_width: int = 128  # Bit Lines

    # --- 精度参数 ---
    enable_quantization: bool = True
    weight_bits: int = 8
    activation_bits: int = 8

    # --- 能耗参数 (单位: pJ) ---

    # MAC 能耗模型: E_mac = Base + (Height * Scaling)
    # 物理规律：阵列越大，位线越长，单次运算能耗越高
    base_mac_pj: float = 0.05  # 极小阵列的基础能耗
    cap_scaling_factor: float = 0.0005  # 每增加一行带来的额外能耗 (寄生电容)

    # 接口能耗
    energy_adc_pj: float = 2.0  # ADC 转换能耗

    # 数字/缓存能耗
    energy_digital_pj: float = 0.05  # 普通数字逻辑 (ReLU/Pool)
    energy_accum_pj: float = (
        0.5  # 部分和累加 (Partial Sum Accumulation): SRAM读写 + 加法器
    )

    @property
    def dynamic_mac_energy(self):
        """
        计算当前阵列尺寸下的单次 MAC 能耗。

        Returns:
            float: 单次乘加运算的能耗 (pJ)。
        """
        return self.base_mac_pj + (self.cim_array_height * self.cap_scaling_factor)


# 全局配置实例
CONFIG = SimConfig()


class StatsRecorder:
    """
    全局统计记录器。

    用于在推理过程中收集各层的硬件统计数据 (MACs, ADC, Latency 等)。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据。"""
        self.layers = defaultdict(
            lambda: {
                "macs": 0,
                "adc_ops": 0,
                "digital_ops": 0,
                "accum_ops": 0,
                "latency_cycles": 0,
            }
        )
        self.current_layer_name = "Unknown"

    def set_layer(self, name):
        """设置当前正在记录的层名称。"""
        self.current_layer_name = name

    def add_macs(self, count):
        self.layers[self.current_layer_name]["macs"] += count

    def add_adc(self, count):
        self.layers[self.current_layer_name]["adc_ops"] += count

    def add_digital(self, count):
        self.layers[self.current_layer_name]["digital_ops"] += count

    def add_accum(self, count):
        self.layers[self.current_layer_name]["accum_ops"] += count

    def add_latency(self, cycles):
        self.layers[self.current_layer_name]["latency_cycles"] += cycles


# 全局统计实例
STATS = StatsRecorder()


def quantize_tensor(x, bits):
    """
    模拟量化操作。

    Args:
        x (Tensor): 输入浮点张量。
        bits (int): 量化位宽。

    Returns:
        Tensor: 量化并反量化后的张量 (Simulated Quantization)。
    """
    if not CONFIG.enable_quantization or bits >= 32:
        return x

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1

    # 简化 Scale 计算 (实际部署通常统计 Dataset 的 min/max)
    abs_max = x.abs().max().item()
    if abs_max == 0:
        return x

    scale = abs_max / qmax
    x_int = (x / scale).round().clamp(qmin, qmax)
    x_recon = x_int * scale
    return x_recon


class CIM_Tile_Sim(nn.Module):
    """
    具备硬件仿真功能的 CIM 阵列模块。

    集成功能：
    - 权重/激活量化模拟。
    - 硬件性能统计 (MACs, Latency, Energy)。
    - 阵列映射 (Mapping) 与分块 (Tiling) 逻辑。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def load_weights(self, original_weight, original_bias=None):
        with torch.no_grad():
            self.weight.copy_(original_weight.view(original_weight.shape[0], -1))
            if original_bias is not None:
                self.bias.copy_(original_bias)

    def forward(self, x_vector):
        """
        执行仿真前向传播。

        Args:
            x_vector (Tensor): [Batch, Length, In_Features]
        """
        if x_vector.dim() == 2:
            x_vector = x_vector.unsqueeze(1)

        batch_size, length, in_dim = x_vector.shape
        out_dim = self.out_features

        # 执行计算 (模拟量化精度)
        x_q = quantize_tensor(x_vector, CONFIG.activation_bits)
        w_q = quantize_tensor(self.weight, CONFIG.weight_bits)
        output = F.linear(x_q, w_q, self.bias)
        output_q = quantize_tensor(output, CONFIG.activation_bits)

        # 性能统计
        with torch.no_grad():
            # A. 基础 MACs
            total_macs = batch_size * length * out_dim * in_dim
            STATS.add_macs(total_macs)

            # B. ADC 次数 (每个输出点一次)
            total_adcs = batch_size * length * out_dim
            STATS.add_adc(total_adcs)

            # C. 阵列映射与分块 (Mapping & Tiling)
            # 计算需要把大矩阵切成多少个小块放入 CIM Array
            h_splits = math.ceil(in_dim / CONFIG.cim_array_width)  # 输入切分
            v_splits = math.ceil(out_dim / CONFIG.cim_array_height)  # 输出切分

            # D. Latency 计算
            # 时间步 = 输入向量数 * 权重分块数
            tile_ops = h_splits * v_splits
            total_input_vectors = batch_size * length
            total_cycles = total_input_vectors * tile_ops
            STATS.add_latency(total_cycles)

            # E. 累加能耗计算 (Accumulation Energy)
            # 如果输入维度超过阵列宽度 (h_splits > 1)，需要跨阵列累加部分和 (Partial Sums)
            # 这是一个非常耗能的操作 (SRAM Read/Write + Digital Add)
            # 累加次数 = 总输出点数 * (切分份数 - 1)
            if h_splits > 1:
                total_output_points = batch_size * length * out_dim
                accum_count = total_output_points * (h_splits - 1)
                STATS.add_accum(accum_count)

        return output_q


class CIM_Conv2d_Sim(nn.Module):
    """
    具备硬件仿真功能的 CIM 卷积层。
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        # Conv2d 转换为矩阵乘法: Input Dim = C * K * K
        self.input_dim = in_channels * kernel_size * kernel_size
        self.cim_tile = CIM_Tile_Sim(self.input_dim, out_channels, bias)

    def forward(self, x):
        n, c, h, w = x.shape

        # im2col 开销
        x_unfolded = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride
        )
        STATS.add_digital(x_unfolded.numel())  # 统计搬运开销

        x_col = x_unfolded.transpose(1, 2)

        # CIM 计算
        out_col = self.cim_tile(x_col)

        # col2im 开销
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out_col.transpose(1, 2).view(n, self.out_channels, h_out, w_out)
        STATS.add_digital(out.numel())

        return out


class CIM_Linear_Sim(nn.Module):
    """
    具备硬件仿真功能的 CIM 全连接层。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.cim_tile = CIM_Tile_Sim(in_features, out_features, bias)

    def forward(self, x):
        return self.cim_tile(x)


class Digital_ReLU(nn.Module):
    """数字 ReLU 层，用于统计数字计算开销。"""

    def forward(self, x):
        STATS.add_digital(x.numel())
        return F.relu(x, inplace=True)


class Digital_MaxPool2d(nn.Module):
    """数字 MaxPool 层，用于统计数字计算开销。"""

    def __init__(self, kernel_size, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        out = self.pool(x)
        STATS.add_digital(out.numel() * 4)  # 估算比较操作次数
        return out


class Digital_Dropout(nn.Module):
    """数字 Dropout 层 (推理时透传)。"""

    def forward(self, x):
        return x


class VGG16_Sim(nn.Module):
    """
    全芯片 VGG16 仿真模型。

    替换所有层为具备统计功能的仿真层。
    """

    def __init__(self, original_model):
        super().__init__()
        self.features = nn.ModuleList()

        # 转换特征提取部分
        layer_idx = 0
        for layer in original_model.features:
            if isinstance(layer, nn.Conv2d):
                name = f"Conv2d_{layer_idx}"
                new_layer = CIM_Conv2d_Sim(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size[0],
                    layer.stride[0],
                    layer.padding[0],
                )
                new_layer.cim_tile.load_weights(layer.weight, layer.bias)
                new_layer._cim_name = name
                self.features.append(new_layer)
                layer_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = f"ReLU_{layer_idx}"
                new_layer = Digital_ReLU()
                new_layer._cim_name = name
                self.features.append(new_layer)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"Pool_{layer_idx}"
                new_layer = Digital_MaxPool2d(2, 2)
                new_layer._cim_name = name
                self.features.append(new_layer)
            else:
                self.features.append(layer)

        self.avgpool = original_model.avgpool

        # 转换分类器部分
        self.classifier = nn.ModuleList()
        for i, layer in enumerate(original_model.classifier):
            if isinstance(layer, nn.Linear):
                name = f"Linear_{i}"
                new_layer = CIM_Linear_Sim(layer.in_features, layer.out_features)
                new_layer.cim_tile.load_weights(layer.weight, layer.bias)
                new_layer._cim_name = name
                self.classifier.append(new_layer)
            elif isinstance(layer, nn.ReLU):
                name = f"Cls_ReLU_{i}"
                new_layer = Digital_ReLU()
                new_layer._cim_name = name
                self.classifier.append(new_layer)
            elif isinstance(layer, nn.Dropout):
                name = f"Dropout_{i}"
                new_layer = Digital_Dropout()
                new_layer._cim_name = name
                self.classifier.append(new_layer)
            else:
                self.classifier.append(layer)

    def forward(self, x):
        STATS.reset()
        for layer in self.features:
            name = getattr(layer, "_cim_name", "Unknown")
            STATS.set_layer(name)
            x = layer(x)

        STATS.set_layer("AvgPool")
        x = self.avgpool(x)
        STATS.add_digital(x.numel())
        x = torch.flatten(x, 1)

        for layer in self.classifier:
            name = getattr(layer, "_cim_name", "Unknown")
            STATS.set_layer(name)
            x = layer(x)
        return x


if __name__ == "__main__":

    print("\n[配置仿真参数]")

    CONFIG.cim_array_height = 256
    CONFIG.cim_array_width = 256

    CONFIG.weight_bits = 8
    CONFIG.activation_bits = 8

    # 打印动态计算出的 MAC 能耗
    current_mac_pj = CONFIG.dynamic_mac_energy

    print(f"    - Array Size: {CONFIG.cim_array_height}x{CONFIG.cim_array_width}")
    print(
        f"    - Dynamic MAC Energy: {current_mac_pj:.4f} pJ (Dependent on Array Height)"
    )
    print(
        f"    - Accumulation Energy: {CONFIG.energy_accum_pj} pJ (Inter-tile communication)"
    )

    print("\n[加载与转换模型]")
    try:
        std_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    except:
        std_vgg = models.vgg16(pretrained=True)

    sim_model = VGG16_Sim(std_vgg)
    sim_model.eval()

    print("\n[运行仿真推理]")
    # 生成随机输入代替真实数据，方便直接运行
    input_tensor = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        sim_output = sim_model(input_tensor)

    print("\n" + "=" * 85)
    print(
        f"{'Layer Name':<12} | {'MACs':<8} | {'Lat(Cyc)':<10} | {'E_MAC':<8} | {'E_Accum':<8} | {'Tot_E(uJ)':<10}"
    )
    print("-" * 85)

    total_macs = 0
    total_energy_pj = 0
    total_latency = 0

    for layer_name, data in STATS.layers.items():
        if data["macs"] == 0 and data["digital_ops"] == 0:
            continue

        # 修正后的总能耗计算公式
        # 1. MAC 能耗 (动态: 阵列大 -> 单次耗电大)
        e_mac = data["macs"] * current_mac_pj

        # 2. 累加能耗 (动态: 阵列小 -> 切分多 -> 累加多 -> 耗电大)
        e_accum = data["accum_ops"] * CONFIG.energy_accum_pj

        # 3. 其他固定能耗
        e_adc = data["adc_ops"] * CONFIG.energy_adc_pj
        e_dig = data["digital_ops"] * CONFIG.energy_digital_pj

        layer_energy = e_mac + e_adc + e_dig + e_accum

        total_macs += data["macs"]
        total_energy_pj += layer_energy
        total_latency += data["latency_cycles"]

        if "Conv" in layer_name or "Linear" in layer_name:
            print(
                f"{layer_name:<12} | {data['macs']/1e6:<8.1f} | {data['latency_cycles']:<10} | "
                f"{e_mac/1e6:<8.2f} | {e_accum/1e6:<8.2f} | {layer_energy/1e6:<10.2f}"
            )

    print("=" * 85)
    print(f"\n📊 芯片总览 (Array: {CONFIG.cim_array_height}x{CONFIG.cim_array_width}):")
    print(f"    - 总延迟 (Latency)     : {total_latency} Cycles")
    print(f"    - 总能耗 (Total Energy): {total_energy_pj/1e6:.4f} uJ")

    if CONFIG.cim_array_height <= 64:
        print("💡 小阵列导致 'E_Accum' (累加能耗) 较高，因为切分次数多。")
    elif CONFIG.cim_array_height >= 512:
        print("💡 大阵列导致 'E_MAC' (计算能耗) 较高，因为位线寄生电容大。")
    else:
        print("💡 这是一个平衡点配置。")
