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
from utils.utils import compute_variable_metrics,gather_by_idx,flatten_dict,split_params,safe_mean
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
        self.gumbel_params, self.other_params = split_params(self.model, keyword="gumbel_selector")

        # 主干（非 gumbel）优化器
        self.opt_main = torch.optim.Adam(
            self.other_params,
            lr=args.lr_main,
            weight_decay=args.wd_main
        )

        # gumbel 优化器（如果存在）
        self.opt_gumbel = None
        if len(self.gumbel_params) > 0:
            self.opt_gumbel = torch.optim.Adam(
                self.gumbel_params,
                lr=args.lr_gumbel,
                weight_decay=args.wd_gumbel
            )

        # 调度器（可选：按需分别设置不同 T_max）
        self.sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_main, T_max=getattr(args, "tmax_main", 50)
        )
        self.sched_gumbel = None
        if self.opt_gumbel is not None:
            self.sched_gumbel = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_gumbel, T_max=getattr(args, "tmax_gumbel", 50)
        )
        self.metric = TemporalWeightedLoss(seq_len=self.args.sample_num-1,main_weight=args.weight_last).to(
            device=self.args.device,
        )
        self.last_metric = compute_variable_metrics
   
    def fit(self):
        self.model = self.model.to(self.args.device)
        for epoch in range(self.args.nepochs):
            train_logs = self._train_one_epoch(epoch_no=epoch)
            val_logs = self.evaluate(epoch_no=epoch)

            # 你可以在这里再统一打印 / 记录 train_logs 和 val_logs
            # 例如：
            if getattr(self.args, "enable_wb", False):
                log_dict = {}
                # 训练 epoch 统计
                for k, v in train_logs.items():
                    if isinstance(v, (int, float)) and v is not None:
                        log_dict[f"train/epoch_{k}"] = v
                # 验证 epoch 统计
                for k, v in val_logs.items():
                    if isinstance(v, (int, float)) and v is not None:
                        log_dict[f"val/epoch_{k}"] = v
                log_dict["epoch"] = epoch
                wandb.log(log_dict)

    def _train_one_epoch(self, epoch_no, disable_tqdm=False):
        # 按 key 聚合 batch -> epoch 的均值
        agg = defaultdict(list)

        with tqdm(self.train_dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                # _train_one_loop 需返回：
                # {
                #   "opt_loss_main": ...,
                #   "opt_loss_gumbel": ... or None,
                #   "last_frame_loss": ...,
                #   "weighted_loss": ...,
                #   "last_metric": dict
                # }
                out = self._train_one_loop(datas)

                # 训练时展示 weighted_loss（主分支优化的 loss）
                show_loss = out["weighted_loss"].item()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=show_loss)

                # 聚合标量
                for k in ["opt_loss_main", "opt_loss_gumbel", "last_frame_loss", "weighted_loss"]:
                    v = out.get(k, None)
                    if v is not None:
                        if torch.is_tensor(v):
                            v = v.item()
                        agg[k].append(v)

                # 记录 last_metric（可能是嵌套 dict）
                last_metric = out["last_metric"]
                if getattr(self.args, "enable_wb", False):
                    # 展平
                    flat_metric = flatten_dict(last_metric, parent_key="train")
                    wandb.log(
                        {
                            "train/batch_opt_loss_main": out["opt_loss_main"].item(),
                            "train/batch_weighted_loss": out["weighted_loss"].item(),
                            "train/batch_last_frame_loss": out["last_frame_loss"].item(),
                            **({"train/batch_opt_loss_gumbel": out["opt_loss_gumbel"].item()} 
                            if out["opt_loss_gumbel"] is not None else {}),
                            **flat_metric,
                            "step": epoch_no * len(self.train_dataset) + idx,
                        }
                    )

        # 计算 epoch 级均值
        epoch_logs = {
            "opt_loss_main": safe_mean(agg["opt_loss_main"]),
            "opt_loss_gumbel": safe_mean(agg["opt_loss_gumbel"]),
            "last_frame_loss": safe_mean(agg["last_frame_loss"]),
            "weighted_loss": safe_mean(agg["weighted_loss"]),
        }
        return epoch_logs

    def _train_one_loop(self, datas):
        out_tensor, start_times_tensor, time_periods_tensor, para_tensor = datas
        x, label = split_train_val(out_tensor)

        self.model.train()

        pre, idx = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
        if idx is not None:
            label = gather_by_idx(label, idx)[:, 1:]

        # 计算两个 loss
        last_frame_loss, weighted_loss = self.metric(label, pre)
        last_metric = self.last_metric(label, pre)

        # ---- 1) 更新 non-Gumbel 参数，用 weighted_loss ----
        self.opt_main.zero_grad(set_to_none=True)
        # 如果还要对 last_frame_loss 对 gumbel 求梯度，则需要保留计算图
        retain = (self.opt_gumbel is not None)
        weighted_loss.backward(retain_graph=retain)
        self.opt_main.step()

        # ---- 2) 仅更新 Gumbel 参数，用 last_frame_loss ----
        if self.opt_gumbel is not None:
            self.opt_gumbel.zero_grad(set_to_none=True)
            g_grads = torch.autograd.grad(
                last_frame_loss,
                self.gumbel_params,
                retain_graph=False,      # 通常这里不再需要图了
                allow_unused=True
            )
            for p, g in zip(self.gumbel_params, g_grads):
                if g is not None:
                    p.grad = g
            self.opt_gumbel.step()

        # ---- step schedulers ----
        if self.sched_main is not None:
            self.sched_main.step()
        if self.sched_gumbel is not None:
            self.sched_gumbel.step()

        return {
            "opt_loss_main": weighted_loss.detach(),
            "opt_loss_gumbel": last_frame_loss.detach() if self.opt_gumbel is not None else None,
            "last_frame_loss": last_frame_loss.detach(),
            "weighted_loss": weighted_loss.detach(),
            "last_metric": last_metric
        }
        
    @torch.no_grad()
    def evaluate(self, epoch_no=0, disable_tqdm=False):
        self.model.eval()

        metric_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        total_weighted_loss = 0.0
        total_last_frame_loss = 0.0
        count = 0

        with tqdm(self.val_dataset, unit="batch", disable=disable_tqdm) as teval:
            for idx, datas in enumerate(teval):
                out_tensor, start_times_tensor, time_periods_tensor, para_tensor = datas
                x, label = split_train_val(out_tensor)

                pre, idxs = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
                if idxs is not None:
                    label = gather_by_idx(label, idxs)[:, 1:]

                # 两个 loss
                last_frame_loss, weighted_loss = self.metric(label, pre)
                total_weighted_loss += weighted_loss.item()
                total_last_frame_loss += last_frame_loss.item()

                last_metric = self.last_metric(label, pre)

                # 累加指标
                for var, parts in last_metric.items():
                    for part_name, values in parts.items():
                        for metric_name, value in values.items():
                            metric_sum[var][part_name][metric_name] += value

                count += 1

                # tqdm 显示
                teval.set_description("Eval")
                teval.set_postfix(loss=total_weighted_loss / count)

        # 取均值
        metric_avg = {}
        for var, parts in metric_sum.items():
            metric_avg[var] = {}
            for part_name, values in parts.items():
                metric_avg[var][part_name] = {k: v / count for k, v in values.items()}

        # 你也可以把 loss 写进 metric_avg 里一并返回
        val_logs = {
            "weighted_loss": total_weighted_loss / count,
            "last_frame_loss": total_last_frame_loss / count,
            **{f"metrics/{k}": v for k, v in flatten_dict(metric_avg).items()}
        }

        # wandb 日志
        if getattr(self.args, "enable_wb", False):
            log_dict = flatten_dict(metric_avg, parent_key="val")
            log_dict["val/epoch_weighted_loss"] = val_logs["weighted_loss"]
            log_dict["val/epoch_last_frame_loss"] = val_logs["last_frame_loss"]
            log_dict["epoch"] = epoch_no
            wandb.log(log_dict)

        return val_logs
