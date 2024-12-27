import matplotlib.pyplot as plt
import numpy as np

# 数据
# data = [
#     [0.6257, 0.7470, 0.7445, 0.6006, 0.2640, 0.7665, 0.3519, 0.2835],
#     [0.6211, 0.735, 0.7316, 0.5907, 0.3137, 0.7501, 0.3629, 0.3744],
#     [0.6243, 0.7349, 0.7316, 0.5926, 0.3048, 0.7489, 0.3639, 0.3215],
#     [0.9839, 0.7881, 0.7869, 0.6226, 0.3241, 0.8085, 0.3871, 0.4234],
#     [0.9516, 0.7633, 0.7638, 0.6028, 0.3294, 0.7868, 0.3715, 0.3779],
#     [0.928, 0.7513, 0.7538, 0.6057, 0.3182, 0.7721, 0.3776, 0.3456],
# ]
# data = [
#     [0.6257, 0.747, 0.7445, 0.6006, 0.264, 0.7665, 0.3519, 0.2835],
#     [0.6211, 0.735, 0.7316, 0.5907, 0.3137, 0.7501, 0.3629, 0.3744],
#     [0.6243, 0.7349, 0.7316, 0.5926, 0.3048, 0.7489, 0.3639, 0.3215],
#     [0.9839, 0.7881, 0.7869, 0.6226, 0.3241, 0.8085, 0.3871, 0.4234],
#     [0.9516, 0.7633, 0.7638, 0.6028, 0.3494, 0.7868, 0.3815, 0.4179],
#     [0.928, 0.7513, 0.7538, 0.6157, 0.3382, 0.7721, 0.3776, 0.3456],
# ]
data=[
    [0.747, 0.7445, 0.6006, 0.264, 0.7665, 0.3519, 0.2835],
    [0.735, 0.7316, 0.5907, 0.3137, 0.7501, 0.3629, 0.3744],
    [0.7349, 0.7316, 0.5926, 0.3048, 0.7489, 0.3639, 0.3215],
    [0.7881, 0.7869, 0.6226, 0.3241, 0.8085, 0.3871, 0.4234],
    [0.7633, 0.7638, 0.6028, 0.3494, 0.7868, 0.3815, 0.4179],
    [0.7513, 0.7538, 0.6157, 0.3382, 0.7721, 0.3776, 0.3456]
]
# 标签
labels = ['G2', 'G3', 'U4', 'P5', 'S6', 'L7', 'E8']
num_vars = len(labels)

# 雷达图函数
def radar_chart(data1, data2, title, ax):
    # 计算角度并闭合多边形
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 数据闭合
    data1 += data1[:1]
    data2 += data2[:1]

    # 设置角度与方向
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # 设置最大值为1，隐藏外边框
    ax.set_ylim(0, 1)
    ax.spines['polar'].set_visible(False)

    # 设置网格与标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="black")
    ax.yaxis.grid(True, linestyle='--', color='black', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', color='black', alpha=0.5)

    # 绘制数据
    ax.plot(angles, data1, linewidth=2, marker='o', label='BC algorithm')
    ax.fill(angles, data1, alpha=0.25)
    ax.plot(angles, data2, linewidth=2, marker='o', label='CGIL algorithm')
    ax.fill(angles, data2, alpha=0.25)

    # 标记底部数值
    bottom_text = f"Row 1: {data1[:-1]}\nRow 2: {data2[:-1]}"
    ax.text(0, -1.2, bottom_text, ha='center', va='center', fontsize=9, color='black')

    # 标题与图例
    ax.set_title(title, size=12, color='black')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

# 绘制多个雷达图
fig, axs = plt.subplots(1, 3, subplot_kw=dict(projection='polar'), figsize=(18, 6))

radar_chart(data[0], data[3], 'Train:Test=9:1', axs[0])
radar_chart(data[1], data[4], 'Train:Test=8:2', axs[1])
radar_chart(data[2], data[5], 'Train:Test=7:3', axs[2])

# 保存图片
plt.tight_layout()
plt.savefig("/home/tianlili/data0/CGIL/IL_model/radar_charts01_00d02.png", dpi=300, bbox_inches='tight')
print("hello,world")
