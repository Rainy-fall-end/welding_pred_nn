# welding_pred_nn
## Dataset
- data_path 应该放入一个文件夹路径，下面有许多子文件夹，子文件夹内有.npz数据文件
- 每次load时会有四个数据：
    - out_tensor,  batch_size,seq_len,I,C,H,W 实际使用时，需要reshape为batch_size,seq_len,I*C,H,W
    - start_times_tensor, batch_size,seq_len
    - time_periods_tensor, batch_size,seq_len
    - para_tensor, batch_size,2
- 对于回归模型，输入为out_tensor[batch_size,0,I,C,H,W],para_tensor,输出为out_tensor[batch_size,-1,I,C,H,W]
