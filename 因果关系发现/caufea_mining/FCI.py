# -*- coding: utf-8 -*-
# @Time : 2023/10/16 19:51
# @Author: Lily Tian
# @FileName: FCI算法.py
# @Software: PyCharm


import pandas as pd
import time
from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci

from causallearn.search.ConstraintBased.FCI import fci
example_data=pd.read_csv("../data/breast_cancer.csv", sep=',', header=None)
data1=example_data.to_numpy() #转换成数组

# default parameters
#g, edges = fci(data1)
# or customized parameters
# g, edges = fci(data1, fisherz, 0.01, verbose=True)
# g, edges = fci(data1,mv_fisherz, 0.01, verbose=True)
g, edges = fci(data1,gsq, 0.01, verbose=True)

from causallearn.utils.GraphUtils import GraphUtils
#获取当前系统时间
current_time=time.strftime("%Y%m%d", time.localtime())
#构建文件名
file_name=current_time
pdy = GraphUtils.to_pydot(g)
pdy.write_png(f'breast_FCI_chisq_0.01_{file_name}.png')




