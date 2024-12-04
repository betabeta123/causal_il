import csv

def count_csv_rows_columns(file_path):
    row_count = 0
    col_count = 0
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_count += 1
            current_col_count = len(row)
            col_count = max(col_count, current_col_count)
    return row_count, col_count
# 指定你的CSV文件路径，这里示例用相对路径，你可以替换成实际的绝对路径
file_path = '/home/tianlili/data0/CGIL/counfac_aug/AUG_upsample_dataset_all.csv'
rows, columns = count_csv_rows_columns(file_path)
print(f"该CSV文件的行数为: {rows}，列数为: {columns}")
    