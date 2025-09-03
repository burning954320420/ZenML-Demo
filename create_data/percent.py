import pandas as pd

def split_csv_chunked(input_file, output_prefix=None, chunk_size=10000):
    """
    分块处理大CSV文件的拆分版本
    """
    # 先计算总行数
    total_rows = sum(1 for line in open(input_file)) - 1  # 减去标题行
    split_point = int(total_rows * 0.8)
    
    if output_prefix is None:
        base_name = input_file.replace('.csv', '')
    else:
        base_name = output_prefix
    
    # 分块读取并写入
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    row_count = 0
    
    with open(f"{base_name}_80_percent.csv", 'w') as f80, \
         open(f"{base_name}_20_percent.csv", 'w') as f20:
        
        header_written = False
        
        for chunk in chunk_iter:
            if not header_written:
                chunk.to_csv(f80, index=False, mode='w')
                chunk.to_csv(f20, index=False, mode='w', header=False)
                header_written = True
                continue
                
            if row_count < split_point:
                chunk.to_csv(f80, index=False, mode='a', header=False)
            else:
                chunk.to_csv(f20, index=False, mode='a', header=False)
                
            row_count += len(chunk)
            
    print(f"文件已拆分完成")

split_csv_chunked('./data/cpu_data_90_day_label.csv', output_prefix='data')