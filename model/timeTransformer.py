import torch
import torch.nn as nn
from layers.vit import SequenceViTAutoencoder
from layers.arImputer import ARImputer
import torch
class E2Epredictor(nn.Module):
    def __init__(self, args,shape):
        super().__init__()
        self.shape = shape
        assert args.C == shape[1]
        self.vit = SequenceViTAutoencoder(
            img_size=(shape[-2],shape[-1]),
            patch_size=(args.H,args.W),
            in_chans=args.C,
            embed_dim=args.embed_dim
        )
        self.ar = ARImputer(
            d = args.embed_dim,
            d_model = args.embed_dim // 2,
            d_ff = args.embed_dim // 2
        )
        
    def forward(self,x):
        (out_tensor, start_times_tensor, time_periods_tensor, para_tensor) = x
        out, pad_info = self.vit.encode(out_tensor) # B,T,D
        out = self.ar(
            x_raw = out,
            start_times = start_times_tensor,
            time_periods = time_periods_tensor,
            para = para_tensor,
        )
        out = self.vit.decode(out, pad_info)
        return out #[4, 43, 20, 71, 49]
    
    @torch.no_grad()
    def evaluate(self,x):
        self.eval()
        return self.forward(x)
