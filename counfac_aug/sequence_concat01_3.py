import os
import pandas as pd

input_folder = '/home/tianlili/data0/CGIL/counfac_aug/output/03noncausal_new_concat'
output_path = '/home/tianlili/data0/CGIL/counfac_aug/output/03noncausal_new_concat/03noncausal_new_concat_all.csv'

def read_csv_files(input_folder, output_path):
    for filename in os.listdir(input_folder):
        # 过滤非 CSV 文件
        if not filename.endswith(".csv"):
            continue
        # 过滤非目标文件
        if not filename.startswith("03noncausal_cancat_sequence_data_"):
            continue
        # 读取 CSV 文件
        file_path = os.path.join(input_folder, filename)
        file_data = pd.read_csv(file_path)
        file_data.to_csv(output_path, mode="a", index=False)
if __name__ == '__main__':
    # 合并文件
    read_csv_files(input_folder, output_path)
    # 读取合并后的结果并检查大小
    data_test = pd.read_csv(output_path)
    print(data_test.shape)
    print(data_test.shape[0] / 7640)




