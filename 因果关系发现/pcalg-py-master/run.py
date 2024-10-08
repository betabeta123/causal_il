import pandas as pd
from src.pc import pc
from src.utils import get_causal_chains, plot

if __name__ == '__main__':
    file_path = '/home/tianlili/data0/AML/LI-Medium_Trans.csv'
    image_path = '/home/tianlili/data0/AML/结果/result_test1.png'
    data = pd.read_csv(file_path)
    n_nodes = data.shape[1]
    labels = data.columns.to_list()

    p = pc(
        suff_stat = { "C": data.corr().values, "n": data.shape[0] },
        verbose = True
    )
    # DFS 因果关系链
    print(get_causal_chains(p, start=3, labels=labels))
    # 画图
    plot(p, labels, image_path)
