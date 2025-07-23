import torch
import torch.nn as nn
from layers.vit import SequenceViTAutoencoder
from layers.arImputer import ARImputer
import torch
from layers.samples import sample_random_with_ends,sample_strided,sample_keep_all,GumbelSelector
class E2Epredictor(nn.Module):
    def __init__(self, args,shape):
        super().__init__()
        self.shape = shape
        assert args.C == shape[1]
        self.vit = SequenceViTAutoencoder(
            args=args,
            img_size=(shape[-2],shape[-1]),
            patch_size=(args.H,args.W),
            in_chans=args.C,
            embed_dim=args.embed_dim
        )
        
        if args.sample == "gumbel":
            self.gumbel_selector = GumbelSelector(
                T=args.step_nums,
                hidden_dim=args.embed_dim // 2,
                k=args.sample_num
            )
        
        self.ar = ARImputer(
            d = args.embed_dim,
            d_model = args.embed_dim // 2,
            d_ff = args.embed_dim // 2
        )
        
        self.sample_fn = self._build_sampler(args)
        
        
    def _build_sampler(self, args):
        if args.sample == "random":
            return lambda _: sample_random_with_ends(args.step_nums, k=args.sample_num)
        elif args.sample == "strided":
            return lambda _: sample_strided(args.step_nums, k=args.sample_num)
        elif args.sample == "none":
            return lambda _: sample_keep_all(args.step_nums)
        elif args.sample == "gumbel":
            # 用已创建的子模块
            return lambda x: self.gumbel_selector(x)
        else:
            raise ValueError(f"Unknown sample mode: {args.sample}")

    
    def _gather_by_idx(self, x, idx):
        """x: (B,T,...)   idx: (B,k)"""
        B, T = x.shape[:2]
        idx_exp = idx.unsqueeze(-1) \
                  .expand(-1, -1, *([1] * (x.dim() - 2)))  # 对除 T 外其余维广播
        return x.gather(1, idx_exp)
    
    def forward(self, x):
        # 解包
        out_tensor, start_times, time_periods, para = x   # shapes 见上文
        B, T = out_tensor.shape[:2]

        first_frame = out_tensor[:, 0]                         # (B,C,H,W)
        ctx = self.vit.encode(first_frame)                    # (B,embed_dim_ctx)

        idx = self.sample_fn((ctx,para))                             # (B,k)

        # 3️⃣ 按相同 idx 对所有序列做 gather
        out_tensor   = self._gather_by_idx(out_tensor,   idx)
        start_times  = self._gather_by_idx(start_times,  idx)
        time_periods = self._gather_by_idx(time_periods, idx)
        para         = para                                    # (B,2)  不随 T 变

        # 4️⃣ 后续编码 → ARImputer → 解码
        z, pad = self.vit.encode(out_tensor)                   # (B,k,D)
        z = self.ar(
            x_raw=z,
            start_times=start_times,
            time_periods=time_periods,
            para=para,
        )
        out = self.vit.decode(z, pad)                          # (B,k,C,H,W)

        return out
    
    @torch.no_grad()
    def evaluate(self,x):
        self.eval()
        return self.forward(x)
