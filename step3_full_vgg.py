"""
全网络 CIM 转换与验证。

本模块将标准 VGG16 模型转换为基于 CIM 的实现。
核心任务包括：
- CIM_Linear: 全连接层的 CIM 实现。
- VGG16_CIM: 完整的 VGG16 模型组装，将 Conv2d 和 Linear 替换为 CIM 版本。
- 端到端验证: 验证转换后的 VGG16 与标准 VGG16 在推理结果上的一致性。

非矩阵计算层（如 ReLU, MaxPool, AvgPool, Dropout）直接复用 PyTorch 原生实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CIM_Tile(nn.Module):
    """
    模拟存内计算 (CIM) 的基础阵列模块。

    该模块是从 Step 2 中复用的基础组件。
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化 CIM 阵列。

        Args:
            in_features (int): 输入特征维度。
            out_features (int): 输出特征维度。
            bias (bool): 是否包含偏置。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def load_weights(self, original_weight, original_bias=None):
        """
        加载权重。

        将卷积核或全连接层的权重统一展平为 2D 矩阵存储。
        """
        with torch.no_grad():
            self.weight.copy_(original_weight.view(original_weight.shape[0], -1))
            if original_bias is not None:
                self.bias.copy_(original_bias)

    def forward(self, x_vector):
        """
        执行矩阵乘法 (Input @ Weight.T + Bias)。
        """
        return F.linear(x_vector, self.weight, self.bias)


class CIM_Conv2d(nn.Module):
    """
    基于 CIM 阵列的 2D 卷积层。
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.input_dim = in_channels * kernel_size * kernel_size
        self.cim_tile = CIM_Tile(self.input_dim, out_channels, bias)

    def forward(self, x):
        n, c, h, w = x.shape

        # im2col: 展开输入图像
        x_unfolded = F.unfold(
            x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride
        )
        x_col = x_unfolded.transpose(1, 2)  # [N, L, Input_Dim]

        # CIM 计算: 矩阵乘法
        out_col = self.cim_tile(x_col)

        # col2im: 恢复空间结构
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = out_col.transpose(1, 2).view(n, self.out_channels, h_out, w_out)
        return out


class CIM_Linear(nn.Module):
    """
    基于 CIM 阵列的全连接层 (Linear Layer)。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 全连接层本质就是一次矩阵乘法，直接复用 CIM_Tile
        self.cim_tile = CIM_Tile(in_features, out_features, bias)

    def forward(self, x):
        """
        全连接层前向传播。

        Args:
            x (Tensor): 输入向量，形状 [Batch, In_Features]。
        """
        return self.cim_tile(x)


class VGG16_CIM(nn.Module):
    """
    CIM 版 VGG16 模型。

    自动遍历标准 VGG16 模型，将 Conv2d 和 Linear 层替换为对应的 CIM 实现，
    并保留原有的 ReLU, MaxPool 等层。
    """

    def __init__(self, original_model):
        """
        初始化 CIM 版 VGG16。

        Args:
            original_model (nn.Module): 预训练的标准 VGG16 模型。
        """
        super().__init__()

        # 转换特征提取部分 (Features)
        self.features = nn.ModuleList()
        for layer in original_model.features:
            if isinstance(layer, nn.Conv2d):
                # 遇到卷积层，替换为 CIM_Conv2d
                new_layer = CIM_Conv2d(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size[0],
                    layer.stride[0],
                    layer.padding[0],
                )
                # 立即从原层加载权重
                new_layer.cim_tile.load_weights(layer.weight, layer.bias)
                self.features.append(new_layer)
            else:
                # 遇到 ReLU, MaxPool 等，直接复用原层
                self.features.append(layer)

        # 保留 AvgPool 层
        self.avgpool = original_model.avgpool

        # 转换分类器部分 (Classifier)
        self.classifier = nn.ModuleList()
        for layer in original_model.classifier:
            if isinstance(layer, nn.Linear):
                # 遇到全连接层，替换为 CIM_Linear
                new_layer = CIM_Linear(layer.in_features, layer.out_features)
                new_layer.cim_tile.load_weights(layer.weight, layer.bias)
                self.classifier.append(new_layer)
            else:
                # 遇到 Dropout, ReLU，直接复用
                self.classifier.append(layer)

    def forward(self, x):
        # 前向传播：串联特征提取层
        for layer in self.features:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平，连接卷积部分和全连接部分

        # 前向传播：串联分类器层
        for layer in self.classifier:
            x = layer(x)

        return x


if __name__ == "__main__":
    print("正在组装 CIM 版 VGG16...")

    # 加载标准预训练模型
    std_vgg = models.vgg16()
    std_weight_path = ".\\model\\vgg16-397923af.pth"
    std_state_dict = torch.load(
        std_weight_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=True,
    )
    std_vgg.load_state_dict(std_state_dict)
    std_vgg.eval()

    # 构建 CIM 模型
    my_cim_vgg = VGG16_CIM(std_vgg)
    my_cim_vgg.eval()  # 确保关闭 Dropout 等训练专用层

    print("模型组装完成。正在加载基准测试数据...")

    # 加载 Step 1 生成的数据
    try:
        data = torch.load('baseline_data.pth')
        input_tensor = data['input']
        target_output = data['output']

        print("开始推理...")
        with torch.no_grad():
            cim_output = my_cim_vgg(input_tensor)

        # 结果比对
        # 获取 Top-5 预测结果
        probs = torch.softmax(cim_output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probs, 5)

        print(f"\n--- 结果验证 ---")
        print(f"CIM 模型预测 Top 5: {top5_catid.numpy()}")

        # 计算与官方结果的误差
        diff = (target_output - cim_output).abs().max().item()
        print(f"与官方模型的最大误差: {diff:.6f}")

        if diff < 1e-4:
            print("\n🎉 完美通过！")
        else:
            print("\n⚠️ 存在误差，请检查 Linear 层的转换是否正确。")

    except FileNotFoundError:
        print("错误：找不到 baseline_data.pth，请先运行 Step 1。")
