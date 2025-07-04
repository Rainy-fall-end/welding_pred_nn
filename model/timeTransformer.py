import torch.nn as nn
from layers.vit import SequenceViTAutoencoder
from layers.arImputer import ARImputer
class E2Epredictor(nn.Module):
    def __init__(self, args,shape):
        super().__init__()
        self.shape = shape
        assert args.C == shape[1] * shape[2]
        self.vit = SequenceViTAutoencoder(
            img_size=(shape[-2],shape[-1]),
            patch_size=(args.H,args.W),
            in_chans=args.C,
            embed_dim=args.embed_dim
        )
        self.ar = ARImputer(
            d = args.embed_dim,
            d_model = args.embde_dim // 2,
            d_ff = args.embde_dim // 2
        )
    def forward(self,x,mask):
        (out_tensor, start_times_tensor, time_periods_tensor, para_tensor) = x
        out = self.vit.encode(out_tensor) # B,T,D
        out = self.ar(
            out_tensor = out,
            start_times_tensor = start_times_tensor,
            time_periods_tensor = time_periods_tensor,
            para_tensor = para_tensor,
            mask = mask
        )
        out = self.vit.decode(out)
        return out
    
