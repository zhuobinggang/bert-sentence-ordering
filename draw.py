import matplotlib.pyplot as plt

def run():
    labels = ['BERT4SO', 'BERSON', 'Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']
    x = [1.0, 7.95, 1.0, 4.0, 4.0, 6.0]
    y = [0.8487, 0.88, 0.8518, 0.8541, 0.8593, 0.8622]

    # 1. 定义属于“我们的手法”的标签
    our_methods = ['Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']

    # 2. 将数据拆分为：基线组（Others）与我们方法组（Ours）
    x_other = [x[i] for i, l in enumerate(labels) if l not in our_methods]
    y_other = [y[i] for i, l in enumerate(labels) if l not in our_methods]
    labels_other = [l for l in labels if l not in our_methods]

    x_our = [x[i] for i, l in enumerate(labels) if l in our_methods]
    y_our = [y[i] for i, l in enumerate(labels) if l in our_methods]
    labels_our = [l for l in labels if l in our_methods]

    fig, ax = plt.subplots()

    ax.grid(True, linestyle='--', alpha=0.5)  # 添加淡淡的网格线增加可读性

    # 3. 绘制基线方法：使用橙色圆圈 (marker='o')，并设置图例标签为 Baselines
    ax.scatter(x_other, y_other, color='#ff7f0e', marker='o', s=100, label='Baselines')

    # 4. 绘制我们的方法：使用蓝色上三角形 (marker='^')，并设置图例标签为 Ours
    ax.scatter(x_our, y_our, color='#1f77b4', marker='^', s=100, label='Ours (Proposed)')

    # 5. 分别为两组添加文字标签，并通过 xytext 调整重叠点的位置
    # 标注基线方法
    for i, txt in enumerate(labels_other):
        xytext = (6, 4)  # 默认向右上方偏移 
        if txt == 'BERT4SO':  # 与 Direct MLM 在 x=1.0 处重叠，将其往下移
            xytext = (6, -12)
        ax.annotate(txt, (x_other[i], y_other[i]), textcoords="offset points", xytext=xytext, fontsize=9)

    # 标注我们的方法
    for i, txt in enumerate(labels_our):
        xytext = (6, 4)  # 默认向右上方偏移
        if txt == 'Top-3 critic':  # 与 Random-2 critic 在 x=4.0 处重叠，将其往下移
            xytext = (6, -12)
        ax.annotate(txt, (x_our[i], y_our[i]), textcoords="offset points", xytext=xytext, fontsize=9)

    # 6. 设置坐标轴、图例与网格
    ax.set_xlabel('Relative FLOPs')
    ax.set_ylabel('Kendall’s Tau ($\tau$)')
    ax.legend()  # 自动根据上文的 label 参数生成图例


    # 保存图片
    plt.savefig('log/scatter_plot_styled.eps', bbox_inches='tight')

def sind():
    labels = ['BERT4SO', 'BERSON', 'BERSON + BOID', 'Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']
    x = [1.0, 7.94, 7.94, 1.0, 4.0, 4.0, 6.0]
    y = [0.5998, 0.65, 0.67, 0.5900, 0.5967, 0.5986, 0.6048]

    # 1. 定义属于“我们的手法”的标签
    our_methods = ['Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']

def double_draw():
    # ==================== 1. 数据准备 ====================
    # ROCStory 数据
    labels_roc = ['BERT4SO', 'BERSON', 'Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']
    x_roc = [1.0, 7.95, 1.0, 4.0, 4.0, 6.0]
    y_roc = [0.8487, 0.88, 0.8518, 0.8541, 0.8593, 0.8622]

    # SIND 数据
    labels_sind = ['BERT4SO', 'BERSON', 'BERSON + BOID', 'Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']
    x_sind = [1.0, 7.94, 7.94, 1.0, 4.0, 4.0, 6.0]
    y_sind = [0.5998, 0.65, 0.67, 0.5900, 0.5967, 0.5986, 0.6048]

    our_methods = ['Direct MLM', 'Top-3 critic', 'Random-2 critic', 'Random-3 critic']


    # ==================== 2. 创建画布 ====================
    # 创建 1 行 2 列的子图布局，sharex=True 共享 X 轴范围，figsize 控制整体宽高比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))


    # ==================== 3. 核心绘图函数 ====================
    def plot_dataset(ax, x_data, y_data, labels_data, dataset_name):
        # 拆分数据组
        x_other = [x_data[i] for i, l in enumerate(labels_data) if l not in our_methods]
        y_other = [y_data[i] for i, l in enumerate(labels_data) if l not in our_methods]
        labels_other = [l for l in labels_data if l not in our_methods]

        x_our = [x_data[i] for i, l in enumerate(labels_data) if l in our_methods]
        y_our = [y_data[i] for i, l in enumerate(labels_data) if l in our_methods]
        labels_our = [l for l in labels_data if l in our_methods]

        # 绘制散点
        ax.scatter(x_other, y_other, color='#ff7f0e', marker='o', s=120, label='Baselines')
        ax.scatter(x_our, y_our, color='#1f77b4', marker='^', s=120, label='Ours (Proposed)')

        # 标注基线方法（针对重叠点做微调）
        for i, txt in enumerate(labels_other):
            xytext = (6, 4)  # 默认位置
            if dataset_name == 'ROCStory' and txt == 'BERT4SO':
                xytext = (6, -14)  # 往下移，避免和 Direct MLM 叠在一起
            elif dataset_name == 'SIND' and txt == 'BERSON':
                xytext = (6, -14)  # 往下移，避免和 BERSON + BOID 叠在一起
            ax.annotate(txt, (x_other[i], y_other[i]), textcoords="offset points", xytext=xytext, fontsize=9.5)

        # 标注我们方法
        for i, txt in enumerate(labels_our):
            xytext = (6, 4)
            if dataset_name == 'ROCStory' and txt == 'Top-3 critic':
                xytext = (6, -14)
            elif dataset_name == 'SIND' and txt == 'Direct MLM':
                xytext = (6, -14)  # SIND 里面 Direct MLM 变低了，将其往下移
            elif dataset_name == 'SIND' and txt == 'Top-3 critic':
                xytext = (6, -14)
            ax.annotate(txt, (x_our[i], y_our[i]), textcoords="offset points", xytext=xytext, fontsize=9.5)

        # 子图基础配置
        ax.set_title(dataset_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Relative FLOPs', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)


    # ==================== 4. 分别绘制左右子图 ====================
    plot_dataset(ax1, x_sind, y_sind, labels_sind, 'SIND')       # 左边放 SIND
    plot_dataset(ax2, x_roc, y_roc, labels_roc, 'ROCStory')   # 右边放 ROCStory

    # Y轴标签只需要加在最左侧的图上，避免视觉重复
    ax1.set_ylabel('Kendall’s Tau (τ)', fontsize=12)
    ax2.set_ylabel('Kendall’s Tau (τ)', fontsize=12) # 如果你希望两个图都有 Y 轴标签可以保留这行，若不需要可以删掉

    # ==================== 5. 提取并合并图例 ====================
    handles, labels = ax1.get_legend_handles_labels()
    # 将图例放在画布正上方 (loc='upper center')，分 2 列展示 (ncol=2)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=11)

    # 自动调整布局，防止边缘裁剪
    plt.tight_layout()

    # 保存为符合 LaTeX 学术规范的 PDF 格式
    plt.savefig('datasets_comparison.eps', bbox_inches='tight')
    print("PDF 图表已成功生成并保存在当前目录下！")