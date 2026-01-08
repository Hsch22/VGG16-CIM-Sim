import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_chart():
    """
    绘制并保存 RTX 3090 vs AIG-CIM 的对比柱状图 (修改版)。
    """
    try:
        # ==========================================
        # 1. 数据准备 (Data Preparation)
        # ==========================================
        categories = ['Stable Diff v2', 'pix2pix']
        x = np.arange(len(categories))
        width = 0.35  # 柱子宽度

        # 数据结构: [Stable Diff v2, pix2pix]
        data = {
            'Latency': {
                'rtx': [24.27, 13.78],
                'cim': [1.14, 2.13],
                # 单位移至标题，此处不再需要后缀
                'unit_label': '(s)',
            },
            'Throughput': {
                'rtx': [0.165, 0.052],
                'cim': [38.15, 19.07],
                'unit_label': '(Task/s)',  # 稍微简写以适应标题
            },
            'Power': {
                'rtx': [321, 318],
                'cim': [96.4, 103],
                'unit_label': '(W)',
            },
        }

        # 配色
        color_rtx = '#d3d3d3'  # 浅灰色
        color_cim = '#faceb3'  # 浅橙色/肉色
        edge_color = 'black'  # 边框颜色

        # ==========================================
        # 2. 绘图设置 (Plot Setup)
        # ==========================================
        plt.style.use('seaborn-v0_8-white')
        fig, axs = plt.subplots(1, 3, figsize=(22, 7))  # 稍微加宽一点画布

        keys = ['Latency', 'Throughput', 'Power']

        # 标题直接包含单位
        titles = [
            f"Latency {data['Latency']['unit_label']}",
            f"Throughput {data['Throughput']['unit_label']}",
            f"Power {data['Power']['unit_label']}",
        ]

        # ==========================================
        # 3. 循环绘制子图 (Plotting Loop)
        # ==========================================
        for i, ax in enumerate(axs):
            key = keys[i]
            d = data[key]
            rtx_vals = d['rtx']
            cim_vals = d['cim']

            # 绘制柱子
            rects1 = ax.bar(
                x - width / 2,
                rtx_vals,
                width,
                label='RTX 3090',
                color=color_rtx,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=3,  # 确保柱子在网格线上方（如果有的话）
            )
            rects2 = ax.bar(
                x + width / 2,
                cim_vals,
                width,
                label='AIG-CIM',
                color=color_cim,
                edgecolor=edge_color,
                linewidth=1.5,
                zorder=3,
            )

            # 设置标题
            # 字体加大，位置左对齐
            ax.set_title(
                titles[i], fontsize=22, fontweight='bold', loc='left', x=-0.02, y=1.02
            )

            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=18, fontweight='bold')

            # --- 坐标轴样式调整 (关键部分) ---

            # 隐藏 上/右 边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # 加粗 下/左 边框
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)

            # 确保左边框从 0 开始
            ax.spines['left'].set_position(
                ('data', -0.5)
            )  # 稍微左移一点，避免紧贴柱子，或者保持默认
            # 这里我们保持默认位置，但要注意箭头对齐

            # 移除 Y 轴原本的刻度数字
            ax.set_yticks([])
            ax.tick_params(axis='x', width=2, length=6)  # 加粗 X 轴刻度线

            # ==========================================
            # 4. 标注数值 (Annotations) - 不带单位
            # ==========================================
            def add_labels(rects):
                for rect in rects:
                    height = rect.get_height()
                    # 动态调整标签位置
                    label_y = height + (max(rtx_vals + cim_vals) * 0.015)
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        label_y,
                        f'{height}',  # 此时只显示数字
                        ha='center',
                        va='bottom',
                        fontsize=18,  # 字体稍大
                        fontweight='bold',
                    )

            add_labels(rects1)
            add_labels(rects2)

            # 设置 Y 轴范围
            max_val = max(rtx_vals + cim_vals)
            ax.set_ylim(0, max_val * 1.2)  # 留出空间

            # 强制 x 轴范围，确保坐标轴看起来比较平衡
            ax.set_xlim(-0.6, 1.6)

        # ==========================================
        # 5. 图例与保存 (Legend & Save)
        # ==========================================
        # 获取图例句柄
        handles, labels = axs[0].get_legend_handles_labels()

        # 放置全局图例
        # fontsize加大, markerscale加大(让方块变大)
        fig.legend(
            handles,
            labels,
            loc='upper center',
            # bbox_to_anchor=(0.5, 1.15),  # 稍微再往上提一点，给大图例留空间
            ncol=2,
            fontsize=24,  # 图例文字变大
            markerscale=2.5,  # 图例色块变大
            frameon=False,  # 无边框
            handlelength=1.5,
            columnspacing=1.5,  # 列间距
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.82)  # 调整顶部空间以容纳大标题和图例

        output_name = 'comparison_gpu_cim.svg'
        plt.savefig(output_name, format='svg', bbox_inches='tight')
        print(f"\n[Output] 结果图表已更新并保存至 {output_name}")

        plt.savefig(
            'comparison_gpu_cim.png',
            format='png',
            dpi=300,
            bbox_inches='tight',
        )
        print(f"[Output] 结果图表已更新并保存至 figure11_comparison_revised.png")

        plt.savefig('comparison_gpu_cim.pdf', format='pdf', bbox_inches='tight')

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\n[Warning] 绘图失败: {e}")


if __name__ == "__main__":
    plot_comparison_chart()
