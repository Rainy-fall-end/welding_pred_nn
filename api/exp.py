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
from utils.utils import compute_variable_metrics,gather_by_idx,flatten_with_slash,split_params,safe_mean
from api.loss import TemporalWeightedLoss
from collections import defaultdict
import torch
from collections import defaultdict
from tqdm import tqdm
class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_dataset, self.val_dataset = build_dataset(args)
        self.input_shape = self.train_dataset.dataset.dataset.shape

        self.model = E2Epredictor(args, self.input_shape).to(device=args.device)

        self.gumbel_params, self.other_params = split_params(
            self.model, keywords="gumbel_selector"
        )

        # --- 优化器 ---
        self.opt_main = torch.optim.Adam(
            self.other_params, lr=args.lr_main, weight_decay=args.wd_main
        )
        self.opt_gumbel = None
        if len(self.gumbel_params) > 0:
            self.opt_gumbel = torch.optim.Adam(
                self.gumbel_params, lr=args.lr_gumbel, weight_decay=args.wd_gumbel
            )

        # --- 调度器（可选） ---
        self.sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_main, T_max=getattr(args, "tmax_main", 50)
        )
        self.sched_gumbel = None
        if self.opt_gumbel is not None:
            self.sched_gumbel = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_gumbel, T_max=getattr(args, "tmax_gumbel", 50)
            )

        # --- 损失与评估 ---
        self.metric = TemporalWeightedLoss(
            seq_len=self.args.sample_num - 1, main_weight=args.weight_last
        ).to(device=self.args.device)
        self.last_metric = compute_variable_metrics  # 你给的 var->{last,overall}->{metrics...}

        # --- wandb 定义（仅在需要时） ---
        if getattr(self.args, "enable_wb", False):
            import wandb
            self.wandb = wandb
            # 仅当你还没有在外部 wandb.init() 时，这里可做一次 init
            if self.wandb.run is None:
                self.wandb.init(project=getattr(args, "wb_project", "default"),
                                name=getattr(args, "wb_name", None),
                                config=vars(args))
            self._wandb_define_metrics()
        else:
            self.wandb = None

        # 最佳验证指标（可选）
        self._best_val = float("inf")

    # -------------------- W&B metrics 定义 --------------------
    def _wandb_define_metrics(self):
        w = self.wandb
        w.define_metric("global_step")
        w.define_metric("epoch")

        # batch 粒度
        w.define_metric("train/batch/*", step_metric="global_step")
        w.define_metric("val/batch/*",   step_metric="global_step")

        # epoch 粒度
        w.define_metric("train/epoch/*", step_metric="epoch")
        w.define_metric("val/epoch/*",   step_metric="epoch")

        # 验证阶段细粒度 metrics（epoch 粒度）
        w.define_metric("val/metrics/*", step_metric="epoch")

        # （可选）记录最优
        w.define_metric("best/*", step_metric="epoch")

    # -------------------- 训练主入口 --------------------
    def fit(self):
        self.model = self.model.to(self.args.device)

        global_step = 0
        steps_per_epoch = len(self.train_dataset)

        for epoch in range(self.args.nepochs):
            # ---- train 一个 epoch（含 batch 级 wandb 日志）
            train_logs = self._train_one_epoch(
                epoch_no=epoch,
                global_step_start=global_step
            )
            global_step += steps_per_epoch

            # ---- 验证（返回 epoch 级 loss + 细粒度 metrics 树）
            val_logs, val_metric_tree = self.evaluate(epoch_no=epoch)

            # ---- W&B：epoch 级日志 + 细粒度 val metrics（flatten 后写）
            if self.wandb is not None:
                flat_val = flatten_with_slash(val_metric_tree, parent_key="val/metrics")

                # 记录当前 epoch 各种指标
                payload = {
                    "epoch": epoch,
                    # train epoch
                    "train/epoch/opt_loss_main": train_logs["opt_loss_main"],
                    "train/epoch/weighted_loss": train_logs["weighted_loss"],
                    "train/epoch/last_frame_loss": train_logs["last_frame_loss"],
                }
                if train_logs["opt_loss_gumbel"] is not None:
                    payload["train/epoch/opt_loss_gumbel"] = train_logs["opt_loss_gumbel"]

                # val epoch
                payload.update({
                    "val/epoch/weighted_loss": val_logs["weighted_loss"],
                    "val/epoch/last_frame_loss": val_logs["last_frame_loss"],
                })

                # 细粒度 val metrics
                payload.update(flat_val)

                self.wandb.log(payload, step=epoch)

                # ---- （可选）写 best
                if val_logs["weighted_loss"] < self._best_val:
                    self._best_val = val_logs["weighted_loss"]
                    # 写 summary
                    self.wandb.run.summary["best/epoch"] = epoch
                    self.wandb.run.summary["best/val/epoch/weighted_loss"] = self._best_val
                    # 也可以把细粒度 metrics 的最好值写进去
                    for k, v in flat_val.items():
                        self.wandb.run.summary[f"best/{k}"] = v

        if self.wandb is not None:
            self.wandb.finish()

    # -------------------- 单个 batch 的训练 --------------------
    def _train_one_loop(self, datas):
        out_tensor, start_times_tensor, time_periods_tensor, para_tensor = datas
        x, label = split_train_val(out_tensor)

        self.model.train()

        pre, idx = self.model((x, start_times_tensor, time_periods_tensor, para_tensor))
        if idx is not None:
            label = gather_by_idx(label, idx)[:, 1:]

        # 两个 loss
        last_frame_loss, weighted_loss = self.metric(label, pre)
        last_metric = self.last_metric(label, pre)

        # ---- 1) 更新 non-Gumbel 参数（weighted_loss）----
        self.opt_main.zero_grad(set_to_none=True)
        retain = (self.opt_gumbel is not None)
        weighted_loss.backward(retain_graph=retain)
        self.opt_main.step()

        # ---- 2) 仅更新 Gumbel 参数（last_frame_loss）----
        if self.opt_gumbel is not None:
            self.opt_gumbel.zero_grad(set_to_none=True)
            g_grads = torch.autograd.grad(
                last_frame_loss,
                self.gumbel_params,
                retain_graph=False,
                allow_unused=True
            )
            for p, g in zip(self.gumbel_params, g_grads):
                if g is not None:
                    p.grad = g
            self.opt_gumbel.step()

        # ---- schedulers ----
        if self.sched_main is not None:
            self.sched_main.step()
        if self.sched_gumbel is not None:
            self.sched_gumbel.step()

        return {
            "opt_loss_main":   weighted_loss.detach(),
            "opt_loss_gumbel": last_frame_loss.detach() if self.opt_gumbel is not None else None,
            "last_frame_loss": last_frame_loss.detach(),
            "weighted_loss":   weighted_loss.detach(),
            "last_metric":     last_metric
        }

    # -------------------- 一个 epoch 的训练（含 batch 日志） --------------------
    def _train_one_epoch(self, epoch_no, global_step_start=0, disable_tqdm=False):
        agg = defaultdict(list)

        with tqdm(self.train_dataset, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, datas in enumerate(tepoch):
                out = self._train_one_loop(datas)

                show_loss = out["weighted_loss"].item()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=show_loss)

                for k in ["opt_loss_main", "opt_loss_gumbel", "last_frame_loss", "weighted_loss"]:
                    v = out.get(k, None)
                    if v is not None:
                        if torch.is_tensor(v):
                            v = v.item()
                        agg[k].append(v)

                # ---- batch 级 wandb.log（使用扁平 key）----
                if self.wandb is not None:
                    base = "train/batch"
                    flat_metric = flatten_with_slash(out["last_metric"], parent_key=f"{base}/metrics")
                    payload = {
                        "global_step": global_step_start + idx,
                        f"{base}/loss/opt_main":   out["opt_loss_main"].item(),
                        f"{base}/loss/weighted":   out["weighted_loss"].item(),
                        f"{base}/loss/last_frame": out["last_frame_loss"].item(),
                        **({f"{base}/loss/gumbel": out["opt_loss_gumbel"].item()}
                           if out["opt_loss_gumbel"] is not None else {}),
                        **flat_metric
                    }
                    self.wandb.log(payload)

        epoch_logs = {
            "opt_loss_main":   safe_mean(agg["opt_loss_main"]),
            "opt_loss_gumbel": safe_mean(agg["opt_loss_gumbel"]),
            "last_frame_loss": safe_mean(agg["last_frame_loss"]),
            "weighted_loss":   safe_mean(agg["weighted_loss"]),
        }
        return epoch_logs

    # -------------------- 验证 --------------------
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

                last_frame_loss, weighted_loss = self.metric(label, pre)
                total_weighted_loss += weighted_loss.item()
                total_last_frame_loss += last_frame_loss.item()

                last_metric = self.last_metric(label, pre)

                # 累加
                for var, parts in last_metric.items():
                    for part_name, values in parts.items():
                        for metric_name, value in values.items():
                            metric_sum[var][part_name][metric_name] += value

                count += 1
                teval.set_description("Eval")
                teval.set_postfix(loss=total_weighted_loss / count)

        # 取均值（保持原树状结构）
        metric_avg = {}
        for var, parts in metric_sum.items():
            metric_avg[var] = {}
            for part_name, values in parts.items():
                metric_avg[var][part_name] = {k: v / count for k, v in values.items()}

        val_logs = {
            "weighted_loss":   total_weighted_loss / count,
            "last_frame_loss": total_last_frame_loss / count,
        }

        return val_logs, metric_avg
