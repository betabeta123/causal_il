# import csv
# import os
# import pandas as pd

# input_folder='/home/tianlili/data0/CGIL/counfac_aug/output/new_concat'
# def read_csv_files(input_folder):
#     for filename in os.listdir(input_folder):
#         print(filename)
#         if not filename.endswith(".csv"):
#             continue
#         # if filename.startswith("AUG_upsample_dataset_0") or filename.startswith("AUG_upsample_dataset_1")  or filename.startswith("AUG_upsample_dataset_2") or filename.startswith("AUG_upsample_dataset_3") or filename.startswith("AUG_upsample_dataset_66.csv"):
#         #     file_data = pd.read_csv(filename)
#         #     print("*",len(file_data))
#         #     file_data.to_csv("AUG_upsample_dataset.csv", mode="a", index=False)
#         if filename.startswith("/home/tianlili/data0/CGIL/counfac_aug/output/new_concat/new2_cancat_sequence_data_"):
#             file_data = pd.read_csv(filename)
#             print(len(file_data))
#         file_data.to_csv("/home/tianlili/data0/CGIL/counfac_aug/output/new_concat/new2_cancat_sequence_data.csv", mode="a", index=False)
# if __name__ == '__main__':
#     # read_csv_files("./")
#     data_test = pd.read_csv("/home/tianlili/data0/CGIL/counfac_aug/output/new_concat/new2_cancat_sequence_data.csv")
#     print(data_test.shape)
#     # print(len(data_test))
#     print((len(data_test)+1)/7640)
import os
import pandas as pd

input_folder = '/home/tianlili/data0/CGIL/counfac_aug/output/new_concat'
output_path = '/home/tianlili/data0/CGIL/counfac_aug/output/new_concat/new2_cancat_sequence_data.csv'

def read_csv_files(input_folder, output_path):
    for filename in os.listdir(input_folder):
        # 过滤非 CSV 文件
        if not filename.endswith(".csv"):
            continue

        # 过滤非目标文件
        if not filename.startswith("new2_cancat_sequence_data_"):
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
