import torch.optim
import wandb
from tqdm import tqdm
from typing import List, Tuple
from dataset.dataloader import build_dataset
from model.timeTransformer import E2Epredictor
import torch
import torch.nn.functional as F
from utils.tensor_convert import (
    split_train_val
)
from utils.utils import compute_variable_metrics,gather_by_idx
from api.loss import TemporalWeightedLoss
from collections import defaultdict
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
        self.metric = TemporalWeightedLoss(seq_len=self.args.sample_num-1).to(
            device=self.args.device,
        )
        self.last_metric = compute_variable_metrics

    def fit(self):
        self.model = self.model.to(self.args.device)
        for epoch in range(self.args.nepochs):
            losses = self._train_one_epoch(epoch_no=epoch)
            loss_avg = torch.stack(losses).mean().item()
            self.evaluate(epoch_no=epoch)
            
    def _train_one_epoch(self, epoch_no, disable_tqdm=False):
        losses = []
        epoch_loss = 0.0
        with tqdm(self.train_dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                loss, last_metric = self._train_one_loop(datas)
                epoch_loss += loss.item()
                losses.append(loss)

                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss / (idx + 1))

                # batch 级别日志
                if getattr(self.args, "enable_wb", False):
                    wandb.log(
                        {"train/batch_loss": loss.item(), **last_metric},
                        step=epoch_no * len(self.train_dataset) + idx,
                    )

        return losses

    def _train_one_loop(self, datas):
        self.optimiser.zero_grad()
        out_tensor, start_times_tensor, time_periods_tensor, para_tensor = datas

        x, label = split_train_val(out_tensor)
        self.model.train()
        pre, idx = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
        label = gather_by_idx(label, idx)[:, 1:]

        loss = self.metric(label, pre)
        last_metric = self.last_metric(label, pre)

        loss.backward()
        self.optimiser.step()
        self.scheduler.step()
        return loss.detach(), last_metric
    @torch.no_grad()
    def evaluate(self, epoch_no=0, disable_tqdm=False):
        self.model.eval()

        metric_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        total_loss = 0.0
        count = 0

        with tqdm(self.val_dataset, unit="batch", disable=disable_tqdm) as teval:
            for idx, datas in enumerate(teval):
                out_tensor, start_times_tensor, time_periods_tensor, para_tensor = datas
                x, label = split_train_val(out_tensor)

                pre, idx = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
                label = gather_by_idx(label, idx)[:, 1:]

                # loss & metrics
                loss = self.metric(label, pre)
                total_loss += loss.item()
                last_metric = self.last_metric(label, pre)

                # 累加指标
                for var, parts in last_metric.items():
                    for part_name, values in parts.items():
                        for metric_name, value in values.items():
                            metric_sum[var][part_name][metric_name] += value

                count += 1

                # tqdm 显示简单标量
                teval.set_description("Eval")
                teval.set_postfix(loss=total_loss / count)

        # 平均指标
        metric_avg = {}
        for var, parts in metric_sum.items():
            metric_avg[var] = {}
            for part_name, values in parts.items():
                metric_avg[var][part_name] = {k: v / count for k, v in values.items()}

        # wandb 展平日志
        if getattr(self.args, "enable_wb", False):
            flat_log = {f"val/{var}_{part}_{m}": v
                        for var, parts in metric_avg.items()
                        for part, metrics in parts.items()
                        for m, v in metrics.items()}
            flat_log["val/epoch_loss"] = total_loss / count
            flat_log["epoch"] = epoch_no
            wandb.log(flat_log)

        return metric_avg
