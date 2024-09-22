import pandas as pd
import numpy as np
from pcalg import estimate_skeleton, estimate_cpdag
import networkx as nx
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv('/home/tianlili/data0/因果关系发现/iris.csv')
data_matrix = data.to_numpy()

# 使用PC算法估计无向图（骨架）
skeleton, sep_sets = estimate_skeleton(data_matrix, alpha=0.05)

# 使用PC算法估计CPDAG（完全部分有向无环图）
cpdag = estimate_cpdag(skeleton, sep_sets)

# 转换为NetworkX图形
graph = nx.DiGraph(cpdag)

# 绘制因果图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_size=5000, node_color='skyblue', font_size=16, font_weight='bold', arrowsize=20)
plt.title('Causal Graph')
plt.show()