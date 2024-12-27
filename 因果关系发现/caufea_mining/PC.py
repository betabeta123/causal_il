# -*- coding: utf-8 -*-            
# @Time : 2023/9/29 22:41
# @Author: Lily Tian
# @FileName: PC.py
# @Software: PyCharm

#这段代码调试不通

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci
from causallearn.search.ConstraintBased.FCI import fci
import matplotlib as plt
from networkx.drawing.nx_pydot import to_pydot
import matplotlib
matplotlib.use('TkAgg')  # 以 'TkAgg' 为例；如有需要，尝试其他后端
import matplotlib.pyplot as plt

example_data=pd.read_csv("../data/data_process4.csv", sep=',', header=None)

#PC、FCI、GFCI因果发现算法

columns = [#23列属性
   'ego_speed',#veh_e行驶速度
    'ego_angle',#veh_e与当前车道中心轴线夹角
    'ego_lane',#veh_e所在车道
    'rel_f0',#veh_e与前车veh_f0的相对距离
    'rel_b1',#veh_e 车与 veh_b1 之间的相对距离
    'rel_f1',#veh_e 车与 veh_f1 之间的相对距离
    'is_safe_f0',#veh_车和veh_f0车在当前速度下是否保持安全距离
    'is_safe_f1',#veh_e车和veh_f1车在当前速度下是否保持安全距离
    'is_safe_b1',#veh_e车和veh_b1车在当前速度下是否保持安全距离
    'f0_speed',#veh_f0车行驶速度
    'f0_acc',#veh_f0车当前加速状况
    'f0_lane',#veh_f0车所在车道
    'f0_intension',#veh_f0车辆驾驶意图
    'b1_speed',#veh_b1车行驶速度
    'b1_acc',#veh_b1车当前加速状况
    'b1_lane',#veh_b1车当前所在车道
    'b1_intension',#veh_b1车驾驶意图
    'f1_speed',#veh_f1车行驶速度
    'f1_acc',#veh_f1车当前加速状况
    'f1_lane',#f1当前所在车道
    'f1_intension',#veh_f1 车辆驾驶意图
    'Weather',#天气
    'decision'#车辆可能采取的驾驶动作，横向纵向合并决策
]

#重命名列名
example_data.columns=columns
#example_data.info()

data1=example_data.to_numpy() #转换成数组


#调用PC算法
example_result = pc(data1, 0.05, gsq, True, 0, -1)

# print(example_result.sepset)
# print(type(example_result.sepset))
# 保存
#np.save('causal_graph_1', example_result.sepset)


# 加载
# causal_graph = np.load('causal_graph_1.npy')
labels={
 0:'ego_speed',
 1:'ego_angle',
 2:'ego_lane',
 3:'rel_f0',
 4:'rel_b1',
 5:'rel_f1',
 6:'is_safe_f0',
 7:'is_safe_f1',
 8:'is_safe_b1',
 9:'f0_speed',
 10:'f0_acc',
 11:'f0_lane',#veh_f0车所在车道
 12:'f0_intension',#veh_f0车辆驾驶意图
 13:'b1_speed',#veh_b1车行驶速度
 14:'b1_acc',#veh_b1车当前加速状况
 15:'b1_lane',#veh_b1车当前所在车道
 16:'b1_intension',#veh_b1车驾驶意图
 17:'f1_speed',#veh_f1车行驶速度
 18:'f1_acc',#veh_f1车当前加速状况
 19:'f1_lane',#f1当前所在车道7
 20:'f1_intension',#veh_f1 车辆驾驶意图
 21:'Weather',#天气
 22:'decision'#车辆可能采取的驾驶动作，横向纵向合并决策
}

#fci
#example_result =fci(example_data, fisherz, 0.01, verbose=True)



example_result.to_nx_graph()
example_result.labels=labels
example_result.draw_nx_graph(skel=False)
print(type(example_result))
print(example_result.G) #打印出图中的节点和边
print(type(example_result.G))
# print("打印example_result.G.dpath")
# print(example_result.G.dpath)
print("打印example_result.G.graph")#入节点为正，出节点为负
print(example_result.G.graph)



#<causallearn.graph.GraphClass.CausalGraph object at 0x000001E076620130>

#example_result.draw_pydot_graph()

# from causallearn.utils.GraphUtils import GraphUtils
# import graphviz
# pgv_g = GraphUtils.to_pgv(example_result[0])
# graphviz.Source(pgv_g)

#加入背景知识
# from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
# nodes=example_result.G.get_nodes()#获取节点列表
# bk=BackgroundKnowledge()\
#  .add_forbidden_by_node(nodes[10],nodes[1])\
#  .add_forbidden_by_node(nodes[10],nodes[2])\
#  .add_forbidden_by_node(nodes[3],nodes[8])
# example_result=pc(data1,0.01,fisherz,True,0,0,background_knowledge=bk)

