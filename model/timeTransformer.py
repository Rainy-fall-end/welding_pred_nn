import torch.nn as nn
from layers.vit import SequenceViTAutoencoder
class E2Epredictor(nn.Module):
    def __init__(self, shape: list[int]=None):
        super().__init__()
        self.vit = SequenceViTAutoencoder()
    
    def forward(self,x):
        (out_tensor, start_times_tensor, time_periods_tensor, para_tensor) = x
        pass