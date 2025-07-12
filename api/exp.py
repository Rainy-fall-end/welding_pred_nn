import torch.optim
import torch.nn as nn
import wandb
from tqdm import tqdm
from typing import List, Tuple
from dataset.dataloader import build_dataset
from model.timeTransformer import E2Epredictor
import random
import torch
import torch.nn.functional as F
from utils.tensor_convert import (
    split_train_val,
    flatten_middle_dimensions,
    unflatten_middle_dimensions,
)
from api.loss import TemporalWeightedLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_dataset, self.val_dataset = build_dataset(args)
        self.input_shape = self.train_dataset.dataset.dataset.shape
        self.model = E2Epredictor(args, self.input_shape).to(
            device=args.device
        )
        self.optimiser = torch.optim.AdamW(
            params=self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimiser, T_max=50
        )
        self.metric = TemporalWeightedLoss(self.input_shape[0] - 1).to(
            self.args.device
        )

    def fit(
        self
    ):
        losses = []
        self.model = self.model.to(self.args.device)
        for epoch in range(self.args.nepochs):
            losses = self._train_one_epoch(
                epoch_no=epoch,
                losses=losses
            )
            loss_avg = (sum(losses) / len(losses)).cpu().numpy()
            accuracy_, accuracy, errors = self.evaluate(epoch_no=epoch)
            wandb.log({"acc": accuracy_})
            wandb.log({"losses": loss_avg})

    def _train_one_epoch(self, epoch_no, losses, disable_tqdm=False):
        epoch_loss = 0
        i = 0
        with tqdm(self.train_dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(datas=datas, losses=losses)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)

                if getattr(self.args, "enable_wb", False):
                    wandb.log(
                        {"train/batch_loss": loss.item()},
                        step=epoch_no * len(self.train_dataset) + idx,
                    )

        if getattr(self.args, "enable_wb", False):
            wandb.log({"train/epoch_loss": (epoch_loss.item() / i), "epoch": epoch_no})

        return losses

    def _train_one_loop(self, datas, losses: List[float]) -> Tuple[float, List[float]]:

        self.optimiser.zero_grad()
        (out_tensor, start_times_tensor, time_periods_tensor, para_tensor) = datas
        out_tensor = flatten_middle_dimensions(out_tensor)
        x, label = split_train_val(out_tensor)
        self.model.train()
        pre = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
        loss = self.metric(label, pre)
        loss.backward()
        self.optimiser.step()
        self.scheduler.step()
        losses.append(loss.detach())
        return loss.detach(), losses

    @torch.no_grad()
    def evaluate(self, epoch_no=0, disable_tqdm=False):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with tqdm(self.val_dataset, unit="batch", disable=disable_tqdm) as teval:
            for idx, datas in enumerate(teval):
                (out_tensor, start_times_tensor, time_periods_tensor, para_tensor) = datas
                out_tensor = flatten_middle_dimensions(out_tensor)
                x, label = split_train_val(out_tensor)

                preds = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
                loss = self.metric(label, preds)
                total_loss += loss.item()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(label.cpu().numpy())

                teval.set_description("Eval")
                teval.set_postfix(loss=total_loss / (idx + 1))

        # 合并所有预测与标签
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_labels, all_preds)

        if getattr(self.args, "enable_wb", False):
            wandb.log({
                "val/mse": mse,
                "val/mae": mae,
                "val/rmse": rmse,
                "val/r2": r2,
                "val/epoch_loss": total_loss / len(self.val_dataset),
                "epoch": epoch_no
            })

        return mse, mae, rmse, r2

    def _cal_mask(
        x: torch.Tensor, mask_ratio_range: tuple[float, float] = (0.2, 1.0)
    ) -> torch.Tensor:
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
