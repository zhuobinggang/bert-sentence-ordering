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
    labels_roc = ['BERT4SO', 'BERSON', 'Direct MLM', 'Random-2', 'Random-3', 'Random-4']
    x_roc = [1.0, 7.95, 1.0, 4.0, 6.0, 8.0]
    y_roc = [0.8457, 0.88, 0.8518, 0.8593, 0.8622, 0.8633]

    # SIND 数据
    labels_sind = ['BERT4SO', 'BERSON', 'BERSON + BOID', 'Direct MLM', 'Random-2', 'Random-3', 'Random-4']
    x_sind = [1.0, 7.94, 7.94, 1.0, 4.0, 6.0, 8.0]
    y_sind = [0.5837, 0.65, 0.67, 0.5900, 0.5986, 0.6048, 0.6054]

    # 我们手法的大类
    our_methods = ['Direct MLM', 'Top-3 critic', 'Random-2', 'Random-3', 'Random-4']

    # 需要用蓝色线连接显示趋势的特定手法列表
    line_methods = ['Direct MLM', 'Random-2', 'Random-3', 'Random-4']


    # ==================== 2. 创建画布 ====================
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

        # ------ 新增：绘制蓝色趋势线 ------
        # 提取需要连线的数据点，并按照 x 坐标升序排序（防止因乱序导致连线交叉）
        line_points = [(x_data[i], y_data[i]) for i, l in enumerate(labels_data) if l in line_methods]
        line_points_sorted = sorted(line_points, key=lambda p: p[0])
        
        if line_points_sorted:
            x_line, y_line = zip(*line_points_sorted)
            # color 使用与 Ours 散点相同的蓝色 (#1f77b4)
            # zorder=1 确保线在散点图标的下面，整体更整洁
            ax.plot(x_line, y_line, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.8, zorder=1)
        # --------------------------------

        # 绘制散点（zorder=2 确保点在线的上方）
        ax.scatter(x_other, y_other, color='#ff7f0e', marker='o', s=120, label='Baselines', zorder=2)
        ax.scatter(x_our, y_our, color='#1f77b4', marker='^', s=120, label='Ours (Proposed)', zorder=2)

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
            # 优化 SIND 和 ROCStory 中 BERT4SO 处于下方的重叠标注
            if txt == 'Direct MLM':
                xytext = (6, 6) # 稍微往上抬，把下方的空间留给基线 BERT4SO
            elif txt == 'Top-3 critic':
                xytext = (6, -14)
            ax.annotate(txt, (x_our[i], y_our[i]), textcoords="offset points", xytext=xytext, fontsize=9.5)

        # 子图基础配置
        ax.set_title(dataset_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Relative FLOPs', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(0.2, 9.5)  # 稍微放宽横轴，防止最右侧文字越界


    # ==================== 4. 分别绘制左右子图 ====================
    plot_dataset(ax1, x_sind, y_sind, labels_sind, 'SIND')       # 左边放 SIND
    plot_dataset(ax2, x_roc, y_roc, labels_roc, 'ROCStory')     # 右边放 ROCStory

    # 为两个图均添加 Y 轴标签（如果你需要的话）
    ax1.set_ylabel('Kendall’s Tau (τ)', fontsize=12)
    ax2.set_ylabel('Kendall’s Tau (τ)', fontsize=12)

    # ==================== 5. 提取并合并图例 ====================
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=11)

    # 自动调整布局，防止边缘裁剪
    plt.tight_layout()

    # 保存为 EPS 格式
    plt.savefig('datasets_comparison.pdf', bbox_inches='tight')
    print("PDF 图表已成功生成并保存在当前目录下！")


def double_draw_critic_strategy():
    # ==================== 1. 数据准备 ====================
    # ROCStory 数据
    labels_roc = ['Direct MLM', 'Top-3', 'Top-5', 'Top-7', 'Random-2', 'Random-3', 'Random-4']
    x_roc = [1.0, 4.0, 6.0, 8.0, 4.0, 6.0, 8.0]
    y_roc = [0.8518, 0.8541, 0.8519, 0.8496, 0.8593, 0.8622, 0.8633]

    # SIND 数据
    labels_sind = ['Direct MLM', 'Top-3', 'Top-5', 'Top-7', 'Random-2', 'Random-3', 'Random-4']
    x_sind = [1.0, 4.0, 6.0, 8.0, 4.0, 6.0, 8.0]
    y_sind = [0.5900, 0.5967, 0.5975, 0.5983, 0.5986, 0.6048, 0.6054]

    # 手法分类与对应的标记符号（Markers）
    o_methods = ['Direct MLM']
    square_methods = ['Top-3', 'Top-5', 'Top-7']
    triangle_methods = ['Random-2', 'Random-3', 'Random-4']

    # 定义两条趋势线各自包含的方法序列（都从 Direct MLM 开始算起）
    top_k_line_series = ['Direct MLM', 'Top-3', 'Top-5', 'Top-7']
    rand_k_line_series = ['Direct MLM', 'Random-2', 'Random-3', 'Random-4']


    # ==================== 2. 创建画布 ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))


    # ==================== 3. 子图绘制函数 ====================
    def plot_dataset(ax, x_data, y_data, labels_data, dataset_name):
        # 根据手法分类拆分散点坐标
        x_o = [x_data[i] for i, l in enumerate(labels_data) if l in o_methods]
        y_o = [y_data[i] for i, l in enumerate(labels_data) if l in o_methods]
        
        x_tri = [x_data[i] for i, l in enumerate(labels_data) if l in triangle_methods]
        y_tri = [y_data[i] for i, l in enumerate(labels_data) if l in triangle_methods]
        
        x_sq = [x_data[i] for i, l in enumerate(labels_data) if l in square_methods]
        y_sq = [y_data[i] for i, l in enumerate(labels_data) if l in square_methods]
        
        # ------ 🔥 新增：绘制两条不同颜色的趋势虚线 ------
        # 1. 绘制 Random-k Critic 趋势虚线（对应蓝色 #1f77b4）
        rand_pts = sorted([(x_data[i], y_data[i]) for i, l in enumerate(labels_data) if l in rand_k_line_series], key=lambda p: p[0])
        x_rand, y_rand = zip(*rand_pts)
        ax.plot(x_rand, y_rand, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        
        # 2. 绘制 Top-k Critic 趋势虚线（对应红色 #d62728）
        top_pts = sorted([(x_data[i], y_data[i]) for i, l in enumerate(labels_data) if l in top_k_line_series], key=lambda p: p[0])
        x_top, y_top = zip(*top_pts)
        ax.plot(x_top, y_top, color='#d62728', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        # ------------------------------------------------
        
        # 绘制三种不同标记与颜色的散点（zorder=2 确保散点盖在虚线上方，不被虚线截断）
        ax.scatter(x_o, y_o, color='#2ca02c', marker='o', s=130, label='Direct MLM', zorder=2)
        ax.scatter(x_tri, y_tri, color='#1f77b4', marker='^', s=130, label='Random-k Critic', zorder=2)
        ax.scatter(x_sq, y_sq, color='#d62728', marker='s', s=130, label='Top-k Critic', zorder=2)
        
        # 遍历并智能标注文字标签（防止重叠）
        for i, txt in enumerate(labels_data):
            xi, yi = x_data[i], y_data[i]
            xytext = (6, 5)  # 默认向右上方偏移
            
            # 针对特定重叠区间进行特异性文字错开处理
            if dataset_name == 'ROCStory':
                if txt == 'Top-3 critic':
                    xytext = (6, -14)  # 往下移，不与高处的 Random-2 冲突
                elif txt == 'Direct MLM':
                    xytext = (-70, 5)  # 将公共起点的文字往左移，视觉上像树枝散开的根部，更美观
            elif dataset_name == 'SIND':
                if txt in ['Top-3 critic', 'Top-5 critic', 'Top-7 critic']:
                    xytext = (6, -14)  # SIND中整个Top系列都偏低，统一向下错开
                elif txt == 'Direct MLM':
                    xytext = (-70, 5)  # 同理往左移
                    
            ax.annotate(txt, (xi, yi), textcoords="offset points", xytext=xytext, fontsize=9.5)
            
        # 各子图基本配置
        ax.set_title(dataset_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Relative FLOPs', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 拓宽 X 轴范围，防止两侧长文本超出图片边界
        ax.set_xlim(0.1, 9.5)


    # ==================== 4. 绘图与布局微调 ====================
    # 分别画左边的 SIND 和右边的 ROCStory
    plot_dataset(ax1, x_sind, y_sind, labels_sind, 'SIND')
    plot_dataset(ax2, x_roc, y_roc, labels_roc, 'ROCStory')

    # 为两个子图均添加 Y 轴标签
    ax1.set_ylabel('Kendall’s Tau (τ)', fontsize=12)
    ax2.set_ylabel('Kendall’s Tau (τ)', fontsize=12)

    # 统一合并提取图例，水平平铺（ncol=3）展示在画布正上方
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=11)

    # 自动紧凑布局
    plt.tight_layout()

    # 保存为符合 LaTeX 导入规范的高清 PDF 矢量图
    plt.savefig('critic_strategies.pdf', bbox_inches='tight')
    print("具有两条策略趋势虚线的高清双子图已成功生成：'critic_strategies.pdf'")