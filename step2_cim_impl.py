"""
存内计算 (CIM) 基础模块实现。

包含以下核心组件：
- CIM_Tile: 模拟存储阵列中的基础矩阵乘法。
- CIM_Conv2d: 基于 im2col 技术实现的 CIM 卷积层。
- 验证脚本: 验证 CIM 实现与标准 PyTorch 卷积层的数值等价性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIM_Tile(nn.Module):
    """
    模拟存内计算 (CIM) 的基础阵列模块。

    在实际硬件中，权重被存储在电阻或电容阵列中，利用物理定律（欧姆定律和基尔霍夫定律）直接在存储位置完成矩阵乘法运算。
    本模块在软件层面模拟这一行为，将计算抽象为矩阵乘法操作。
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化 CIM 阵列。

        Args:
            in_features (int): 输入特征维度（对应阵列的列数或输入线）。
            out_features (int): 输出特征维度（对应阵列的行数或输出线）。
            bias (bool): 是否包含偏置项。
        """
        super().__init__()
        # 权重矩阵形状: [out_features, in_features]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def load_weights(self, original_weight, original_bias=None):
        """
        从标准模型加载权重到 CIM 模拟器中。

        Args:
            original_weight (Tensor): 原始卷积层或全连接层的权重。
                对于卷积层，形状为 [Out_Channel, In_Channel, K, K]。
            original_bias (Tensor, optional): 原始偏置项。
        """
        with torch.no_grad():
            # 展平权重：将 4D 卷积核转换为 2D 矩阵以适应 CIM 阵列结构
            # view(out_channels, -1) 将后续维度合并，形状变为 [Out_Channel, Input_Dim]
            self.weight.copy_(original_weight.view(original_weight.shape[0], -1))

            if original_bias is not None:
                self.bias.copy_(original_bias)

    def forward(self, x_vector):
        """
        执行前向计算。

        Args:
            x_vector (Tensor): 输入向量，形状为 [Batch, Length, In_Features]。

        Returns:
            Tensor: 输出结果，执行 X * W^T + Bias 运算。
        """
        # Linear 操作等价于 x @ self.weight.t() + self.bias
        # 这里模拟的是纯粹的矩阵向量乘法 (MVM)，不包含卷积的滑动窗口逻辑
        return F.linear(x_vector, self.weight, self.bias)


class CIM_Conv2d(nn.Module):
    """
    基于存内计算阵列实现的 2D 卷积层。

    通过 im2col (image to column) 技术将卷积操作转换为矩阵乘法，
    从而利用 CIM_Tile 完成计算。
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 计算展开后的输入特征维度：In_Channel * K * K
        self.input_dim = in_channels * kernel_size * kernel_size

        # 实例化底层的存算阵列
        self.cim_tile = CIM_Tile(self.input_dim, out_channels, bias)

    def forward(self, x):
        """
        卷积层前向传播。

        Args:
            x (Tensor): 输入图像，形状 [Batch, Channel, Height, Width]。

        Returns:
            Tensor: 卷积输出，形状 [Batch, Out_Channel, H_out, W_out]。
        """
        # x: [Batch, Channel, Height, Width]
        n, c, h, w = x.shape

        # im2col (数据重排)
        # 将卷积转换为矩阵乘法
        # unfold 将图片中每一个 k*k 的小窗提取出来拉成一条线
        x_unfolded = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        # x_unfolded shape: [N, C*K*K, L]  (L 是滑窗的总数量)

        # 调整维度以适配矩阵乘法：[N, L, Input_Dim]
        x_col = x_unfolded.transpose(1, 2)

        # 调用 CIM 硬件进行计算
        out_col = self.cim_tile(x_col)
        # out_col shape: [N, L, Out_Channels]

        # 计算输出特征图的尺寸
        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 此时 out_col 是 [N, L, Out_Channels]
        # 把它变回 [N, Out_Channels, H_out, W_out]

        # 交换维度 -> [N, Out_Channels, L]
        out_col = out_col.transpose(1, 2)

        # col2im: 折叠回图片形状
        out = out_col.view(n, self.out_channels, h_out, w_out)

        return out


if __name__ == "__main__":
    # 设置随机种子，保证每次运行结果一致
    torch.manual_seed(42)

    # 定义测试参数
    bs, in_c, out_c = 2, 64, 128
    h, w = 32, 32
    k, s, p = 3, 1, 1

    # 创建随机输入数据
    dummy_input = torch.randn(bs, in_c, h, w)

    # 创建标准 Conv2d (Ground Truth)
    std_conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)

    # 创建 CIM_Conv2d (待测对象)
    my_cim_conv = CIM_Conv2d(in_c, out_c, k, stride=s, padding=p)

    # 将标准层的权重加载到 CIM 模块中
    print("正在加载权重...")
    my_cim_conv.cim_tile.load_weights(std_conv.weight, std_conv.bias)

    # 分别运行两次推理
    with torch.no_grad():
        out_std = std_conv(dummy_input)
        out_cim = my_cim_conv(dummy_input)

    # 计算数值误差
    diff = (out_std - out_cim).abs().max().item()

    print(f"\n验证结果:")
    print(f"标准层输出形状: {out_std.shape}")
    print(f"CIM 层输出形状: {out_cim.shape}")
    print(f"最大绝对误差: {diff:.8f}")

    if diff < 1e-5:
        print("\n✅ 成功！CIM 卷积实现与标准卷积数学上等价。")
    else:
        print("\n❌ 警告！误差过大，请检查 im2col 或权重映射逻辑。")
