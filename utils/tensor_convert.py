import torch
def flatten_middle_dimensions(tensor):
    B, T, I, C, H, W = tensor.shape
    return tensor.view(B, T , I * C, H, W)

def unflatten_middle_dimensions(tensor):
    """
    将 shape = (B, T*2*10, H, W) 的张量还原为 (B, T, 2, 10, H, W)

    :param tensor: 输入张量，shape = (B, N=T*2*10, H, W)
    :param T: 原始的时间步数
    :return: 还原后的张量，shape = (B, T, 2, 10, H, W)
    """
    B, T, N, H, W = tensor.shape
    return tensor.view(B, T, 2, 10, H, W)

def unflatten_middle_dimensions_numpy(tensor):
    """
    将 shape = (B, T*2*10, H, W) 的张量还原为 (B, T, 2, 10, H, W)

    :param tensor: 输入张量，shape = (B, N=T*2*10, H, W)
    :param T: 原始的时间步数
    :return: 还原后的张量，shape = (B, T, 2, 10, H, W)
    """
    B, T, N, H, W = tensor.shape
    return tensor.view(B, T, 2, 10, H, W)
def tensor_to_pointcloud_dict(tensor: torch.Tensor) -> dict:
    """
    将 shape 为 (T, 2, 10, H, W) 的 tensor 解析为点云字典结构。
    每个字段 reshape 为 (N, d)，其中 N = H * W。
    """
    T, num_inst, F, H, W = tensor.shape
    assert F == 10, "feature dim must be 10 = u(3) + s(6) + nt(1)"
    N = H * W

    result = {}
    for t in range(T):
        result[t] = {}
        for inst in range(num_inst):
            feat = tensor[t, inst]  # (10, H, W)
            feat = feat.view(F, -1).transpose(0, 1)  # → (N, 10)

            u = feat[:, 0:3]   # (N, 3)
            s = feat[:, 3:9]   # (N, 6)
            nt = feat[:, 9]    # (N,)

            result[t][inst] = {
                "u": u,
                "s": s,
                "nt": nt
            }
    return result

def split_train_val(tensor: torch.Tensor) -> dict:
    return tensor[:],tensor[:]

