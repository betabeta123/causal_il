# -*- coding: utf-8 -*-
# @Time : 2023/10/29 17:12
# @Author: Lily Tian
# @FileName: pc_algorithm.py
# @Software: PyCharm

#这段代码调试不通
import time
from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci


# default parameters
# cg = pc(data)
#example_data=pd.read_csv("../data/data_process4.csv", sep=',', header=None)
example_data=pd.read_csv("../data/data_process66.csv", sep=',', header=None)
data1=example_data.to_numpy() #转换成数组
# or customized parameters
#cg = pc(data1, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)


# cg = pc(data1, 0.05, fisherz, True, 0, -1)# Fisher Z 条件独立性检验
# cg = pc(data1, 0.05, chisq, True, 0, -1)# 卡方独立性检验
# cg = pc(data1, 0.05, gsq, True, 0, -1)# G方检验
# cg = pc(data1, 0.05, kci, True, 0, -1)# kci核方法的独立性检验
cg = pc(data1, 0.01, fisherz, True, 0, -1)# 缺失值 Fisher Z 条件独立性检验

# visualization using pydot
cg.draw_pydot_graph()


# or save the graph
from causallearn.utils.GraphUtils import GraphUtils

#获取当前系统时间
current_time=time.strftime("%Y%m%d", time.localtime())
#构造文件名
#file_name=current_time+"pc.png"
file_name=current_time

pyd = GraphUtils.to_pydot(cg.G)
pyd.write_png(f'PC_fisherz_0.01_{file_name}.png')

# visualization using networkx
# cg.to_nx_graph()
# cg.draw_nx_graph(skel=False)