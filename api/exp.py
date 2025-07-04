import torch.optim
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List,Tuple
from dataset.dataloader import build_dataset
from model.timeTransformer import E2Epredictor
import random
import torch
import torch.nn.functional as F
class Trainer:
    def __init__(
        self,
        args
    ):
        self.args = args
        self.dataset = build_dataset(args)
        self.model = E2Epredictor(args,self.dataset.shape)
        self.optimiser = torch.optim.AdamW(params=self.model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimiser, T_max=50)

    def fit(self, dataloader: torch.utils.data.DataLoader, model: nn.Module, optimiser: torch.optim.Optimizer,scheduler,test_data):
        losses = []
        model = model.to(self.device)
        model.train()
        model.double()
        for epoch in range(self.n_epochs):
            losses = self.train_one_epoch(dataloader=dataloader, epoch_no=epoch, losses=losses, optimiser=optimiser, model=model,scheduler=scheduler)
            loss_avg = (sum(losses)/len(losses)).cpu().numpy()
            accuracy_,accuracy,errors = self.evaluate(dataloader=test_data,model=model)
            wandb.log({"acc": accuracy_})
            wandb.log({"losses": loss_avg})

    def train_one_epoch(self, epoch_no, losses, disable_tqdm=False):
        epoch_loss = 0
        i = 0
        with tqdm(self.dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(label=datas, losses=losses)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)

                if getattr(self.args, "enable_wb", False):
                    wandb.log({"train/batch_loss": loss.item()}, step=epoch_no * len(self.dataset) + idx)

        if getattr(self.args, "enable_wb", False):
            wandb.log({"train/epoch_loss": (epoch_loss.item() / i), "epoch": epoch_no})

        return losses



    def _train_one_loop(
        self, label: torch.utils.data.DataLoader, losses: List[float]) -> Tuple[float, List[float]]:

        self.optimiser.zero_grad()
        mask = self._cal_mask(label[0])
        pre = self.model(label,mask)
        loss = self._cal_loss(
            label=label[0],
            output=pre,
            mask=mask,
            weight_last=self.args.weight_last
        )
        loss.backward()
        self.optimiser.step()
        self.scheduler.step()
        losses.append(loss.detach())
        return loss.detach(), losses

    def evaluate(self, dataloader: torch.utils.data.DataLoader, model: nn.Module):
        """Run the model on the test set and return the accuracy."""
        model.eval()
        
    def _cal_mask(x: torch.Tensor, mask_ratio_range: tuple[float, float]=(0.2,1.0)) -> torch.Tensor:
        B, S, _ = x.shape
        mask = torch.zeros(B, S, dtype=torch.bool)

        for i in range(B):
            mask_ratio = random.uniform(*mask_ratio_range)
            keep_len = int(S * (1 - mask_ratio))
            keep_len = max(1, keep_len)  # 至少保留一个
            # 构造 mask
            mask_1d = torch.zeros(S, dtype=torch.bool)
            mask_1d[:keep_len] = True
            mask_1d[0] = True  # 强制保留第一个时间步
            mask[i] = mask_1d

        return mask
    
    def _cal_loss(self,label: torch.Tensor, output: torch.Tensor, mask: torch.Tensor, weight_last: float = 2.0) -> torch.Tensor:
        B, S, D = label.shape

        weights = torch.ones_like(mask, dtype=label.dtype)
        weights[:, -1] = weight_last

        valid_mask = ~mask  
        weights = weights * valid_mask 

        weights = weights.unsqueeze(-1)  # (B, S, 1)

        squared_error = (output - label) ** 2  # (B, S, D)
        weighted_error = squared_error * weights 

        total_weight = weights.sum()
        if total_weight == 0:
            return torch.tensor(0.0, device=label.device)

        loss = weighted_error.sum() / total_weight
        return loss

