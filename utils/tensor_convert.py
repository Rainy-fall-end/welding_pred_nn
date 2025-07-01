def flatten_middle_dimensions(tensor):
    """
    将 shape 为 (B, T, 2, 10, 71, 49) 的张量重塑为 (B, T*2*10, 71, 49)

    :param tensor: torch.Tensor，shape = (B, T, 2, 10, 71, 49)
    :return: torch.Tensor，shape = (B, T*2*10, 71, 49)
    """
    B, T, I, C, H, W = tensor.shape
    return tensor.view(B, T , I * C, H, W)

def unflatten_middle_dimensions(tensor, T=44):
    """
    将 shape = (B, T*2*10, H, W) 的张量还原为 (B, T, 2, 10, H, W)

    :param tensor: 输入张量，shape = (B, N=T*2*10, H, W)
    :param T: 原始的时间步数
    :return: 还原后的张量，shape = (B, T, 2, 10, H, W)
    """
    B, T, N, H, W = tensor.shape
    return tensor.view(B, T, 2, 10, H, W)
