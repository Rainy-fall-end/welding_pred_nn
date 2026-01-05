# ppo_fem.py
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
from rl.abqs import run_one_job  # run_one_job(run_id, args, current_voltage_seq) -> (success, outdir, reward)
import matplotlib.pyplot as plt
import wandb

# --- gym / gymnasium 兼容导入 ---
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    GYMNASIUM = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PPOConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练轮次：一次 rollout = rollout_steps 次仿真（每次仿真一次 Abaqus）
    total_timesteps: int = 2000
    rollout_steps: int = 20
    num_minibatches: int = 4
    update_epochs: int = 10

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 1e-1
    max_grad_norm: float = 0.5

    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0  # Abaqus 黑盒任务一般熵项可先关掉/很小

    target_kl: float = 0.03

    normalize_adv: bool = True
    anneal_lr: bool = True

    # 你的控制时域：输出 t=0..45 共 46 个点
    t_max: int = 45

    # 失败时给一个保底 reward（如果 pipeline 没返回 reward）
    fail_reward: float = -1e6


class SquashedGaussianActorCritic(nn.Module):
    """
    连续动作：tanh-squashed Gaussian policy（动作范围 [-1,1]）
    后续会映射到 [wu_min,wu_max] / [wi_min,wi_max]
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

        # 每个维度一个 log_std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        return mu, value

    def _dist(self, mu: torch.Tensor):
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @staticmethod
    def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, value = self.forward(obs)
        dist = self._dist(mu)

        raw = dist.rsample()  # reparameterization
        action = torch.tanh(raw)

        # logprob with tanh correction
        logprob_raw = dist.log_prob(raw).sum(-1)
        correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        logprob = logprob_raw - correction

        return action, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mu, value = self.forward(obs)
        dist = self._dist(mu)

        # invert tanh: raw = atanh(action)
        raw = self._atanh(actions)
        logprob_raw = dist.log_prob(raw).sum(-1)
        correction = torch.log(1 - actions.pow(2) + 1e-6).sum(-1)
        logprob = logprob_raw - correction

        entropy = dist.entropy().sum(-1)  # 近似即可（很多实现也这么做）
        return logprob, entropy, value


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        act_dim: int,
        device: str,
        wandb_run=None,
        log_prefix: str = "buffer",
    ):
        self.rollout_steps = rollout_steps
        self.device = device

        self.obs = torch.zeros((rollout_steps, obs_dim), device=device)
        self.actions = torch.zeros((rollout_steps, act_dim), device=device, dtype=torch.float32)
        self.logprobs = torch.zeros((rollout_steps,), device=device)
        self.rewards = torch.zeros((rollout_steps,), device=device)
        self.dones = torch.zeros((rollout_steps,), device=device)
        self.values = torch.zeros((rollout_steps,), device=device)

        self.advantages = torch.zeros((rollout_steps,), device=device)
        self.returns = torch.zeros((rollout_steps,), device=device)

        self.ptr = 0

        # wandb
        self.wandb_run = wandb_run
        self.log_prefix = log_prefix

    def _log(self, data: Dict, step: Optional[int] = None) -> None:
        if self.wandb_run is None:
            return
        # step=None 时 wandb 会用内部 step；建议外部统一传 global_step
        self.wandb_run.log(data, step=step)

    def add(self, obs, action, logprob, reward, done, value, global_step: Optional[int] = None):
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.ptr += 1

        # 轻量日志：buffer 填充进度 + 单步 reward（可按需关掉）
        self._log(
            {
                f"{self.log_prefix}/ptr": self.ptr,
                f"{self.log_prefix}/fill_ratio": self.ptr / float(self.rollout_steps),
                f"{self.log_prefix}/reward_step": float(reward),
                f"{self.log_prefix}/done_step": float(done),
            },
            step=global_step,
        )

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float, global_step: Optional[int] = None) -> None:
        adv = 0.0
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_non_terminal * next_value - self.values[t]
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

        # 统计日志：GAE/return 分布（训练很有用）
        adv_cpu = self.advantages.detach().float()
        ret_cpu = self.returns.detach().float()
        rew_cpu = self.rewards.detach().float()

        self._log(
            {
                f"{self.log_prefix}/reward_mean": float(rew_cpu.mean().item()),
                f"{self.log_prefix}/reward_std": float(rew_cpu.std(unbiased=False).item()),
                f"{self.log_prefix}/adv_mean": float(adv_cpu.mean().item()),
                f"{self.log_prefix}/adv_std": float(adv_cpu.std(unbiased=False).item()),
                f"{self.log_prefix}/return_mean": float(ret_cpu.mean().item()),
                f"{self.log_prefix}/return_std": float(ret_cpu.std(unbiased=False).item()),
            },
            step=global_step,
        )

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.ptr == self.rollout_steps
        return {
            "obs": self.obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values,
        }

    def reset(self, global_step: Optional[int] = None) -> None:
        # 可选：记录一次 reset
        self.ptr = 0
        self._log({f"{self.log_prefix}/reset": 1}, step=global_step)


class FEMOneShotEnv(gym.Env):
    """
    一个 episode = 一次 Abaqus 仿真（one-shot black-box optimization）
    - action: shape=(T*2,) 先在 [-1,1]，再映射到 (wu, wi) 的取值范围
    - step(action): 调用 run_one_job(...) 返回 reward
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        args: argparse.Namespace,
        cfg,
        wandb_run=None,
        log_prefix: str = "env",
        plot_interval: Optional[int] = None,   # 每隔多少次 run_id 画一次序列
    ):
        super().__init__()
        self.args = args
        self.cfg = cfg

        self.T = cfg.t_max + 1
        self.act_dim = self.T * 2

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)

        self._run_id = 0

        # wandb
        self.wandb_run = wandb_run
        self.log_prefix = log_prefix
        self.plot_interval = plot_interval if plot_interval is not None else getattr(cfg, "wandb_plot_interval", 1)

    def _log(self, data: dict, step: Optional[int] = None) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(data, step=step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros((1,), dtype=np.float32)
        info = {}
        return (obs, info) if "gymnasium" in str(type(self)).lower() else obs  # 保留你的兼容逻辑也可

    def _map_to_range(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return lo + (x + 1.0) * 0.5 * (hi - lo)

    def _action_to_seq(self, action: np.ndarray) -> List[Tuple[int, float, float]]:
        action = action.reshape(self.T, 2)
        wu = self._map_to_range(action[:, 0], self.args.wu_min, self.args.wu_max)
        wi = self._map_to_range(action[:, 1], self.args.wi_min, self.args.wi_max)
        return [(t, float(wu[t]), float(wi[t])) for t in range(self.T)]

    def _plot_wu_wi(self, current_voltage_seq: List[Tuple[int, float, float]]):
        # current_voltage_seq: [(t, wu, wi), ...]
        t = [x[0] for x in current_voltage_seq]
        wu = [x[1] for x in current_voltage_seq]
        wi = [x[2] for x in current_voltage_seq]

        fig = plt.figure()
        plt.plot(t, wu, label="wu (voltage)")
        plt.plot(t, wi, label="wi (current)")
        plt.xlabel("t")
        plt.ylabel("value")
        plt.title("Learned voltage/current schedule (stepwise)")
        plt.legend()
        plt.tight_layout()
        return fig

    def step(self, action, global_step: Optional[int] = None):
        # 确保动作合法
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_voltage_seq = self._action_to_seq(action)

        success, outdir, reward = run_one_job(
            run_id=self._run_id,
            args=self.args,
            current_voltage_seq=current_voltage_seq,
        )

        raw_reward = reward
        if reward is None:
            reward_val = float(self.cfg.fail_reward)
        else:
            raw_reward[0] *= self.args.w_e
            raw_reward[1] *= self.args.w_u
            raw_reward[2] *= self.args.w_s
            # 如果 reward 是 (e,u,s) 这种，记录分量并转换成标量
            if isinstance(reward, (list, tuple, np.ndarray)) and len(reward) >= 3:
                self._log(
                    {
                        f"{self.log_prefix}/reward_e": float(reward[0]),
                        f"{self.log_prefix}/reward_u": float(reward[1]),
                        f"{self.log_prefix}/reward_s": float(reward[2]),
                    },
                    step=global_step,
                )
                # 你这里原本 float(reward) 会报错；必须明确怎么合成
                # 若你已有合成规则，把下面这一行改成你的规则
                reward_val = float(np.sum(np.asarray(reward, dtype=np.float64)))
            else:
                reward_val = float(reward)

        # 记录基础标量日志
        self._log(
            {
                f"{self.log_prefix}/run_id": int(self._run_id),
                f"{self.log_prefix}/success": float(bool(success)),
                f"{self.log_prefix}/reward": float(reward_val),
            },
            step=global_step,
        )

        # 定期画“模型学出来的电压/电流序列”
        # 这里用 run_id 做间隔控制；也可以改成 global_step 间隔
        if self.wandb_run is not None and self.plot_interval and (self._run_id % int(self.plot_interval) == 0):
            fig = self._plot_wu_wi(current_voltage_seq)
            self._log(
                {
                    f"{self.log_prefix}/wu_wi_curve": wandb.Image(fig),
                    # 可选：把序列原始值也存下来（便于后处理）
                    f"{self.log_prefix}/wu_mean": float(np.mean([x[1] for x in current_voltage_seq])),
                    f"{self.log_prefix}/wi_mean": float(np.mean([x[2] for x in current_voltage_seq])),
                },
                step=global_step,
            )
            plt.close(fig)

        self._run_id += 1

        # one-shot：一步结束
        obs = np.zeros((1,), dtype=np.float32)
        info = {"success": success, "outdir": outdir, "raw_reward": raw_reward}

        terminated = True
        truncated = False

        # 兼容 gymnasium / gym：你原来用 GYMNASIUM 标志也可以沿用
        try:
            import gymnasium  # noqa: F401
            # 如果你实际是 gymnasium，按 gymnasium 返回
            return obs, float(reward_val), terminated, truncated, info
        except Exception:
            done = True
            return obs, float(reward_val), done, info


def build_batch_args() -> argparse.Namespace:
    """
    复用 batch_runner.py 的参数风格（这里给默认值；你也可以改成从命令行读取）
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job_prefix", default="Plate4Job_")

    parser.add_argument("--inp", default="abaqus_data/workdir_pipeline/Job-4.inp")
    parser.add_argument("--part", default="P4_19")
    parser.add_argument("--delta", default="0.0004")
    parser.add_argument("--wu_min", type=float, default=10)
    parser.add_argument("--wu_max", type=float, default=15)
    parser.add_argument("--wi_min", type=float, default=35)
    parser.add_argument("--wi_max", type=float, default=50)
    parser.add_argument("--fortran", default="scripts/Plate4.for")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--export_script", default="abaqus_data/export_all_data.py")
    parser.add_argument("--cmd", default="abaqus")
    parser.add_argument("--cmd_path", default=r"C:\apps\engine\abaqus2022\commands\abaqus.bat")
    parser.add_argument("--t", type=int, default=5)
    parser.add_argument("--w_e", default=1)
    parser.add_argument("--w_u", default=1e+3)
    parser.add_argument("--w_s", default=1e-8)

    # batch_runner 里还有 max_runs/delay，但 RL 训练里不靠它；仍保留以便 run_one_job 内 sleep 使用
    parser.add_argument("--delay", type=int, default=0)

    args, _ = parser.parse_known_args()
    return args

def ppo_train(cfg: "PPOConfig", args: argparse.Namespace, wandb_run=None) -> None:
    """
    PPO 训练主循环（one-shot env）
    - wandb_run 由外部 wandb.init(...) 传入；此处不做 init
    - 记录训练标量日志 + 每隔 plot_interval 记录一次 wu/wi 曲线（由 env 内部完成）
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- Env + Buffer with wandb logging ----
    env = FEMOneShotEnv(
        args=args,
        cfg=cfg,
        wandb_run=wandb_run,
        log_prefix="env",
        plot_interval=getattr(cfg, "wandb_plot_interval", 1),  # 每隔多少次仿真画一次曲线
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = SquashedGaussianActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    buffer = RolloutBuffer(
        cfg.rollout_steps,
        obs_dim,
        act_dim,
        cfg.device,
        wandb_run=wandb_run,
        log_prefix="buffer",
    )

    # 初始 obs
    if GYMNASIUM:
        obs = env.reset(seed=cfg.seed)
    else:
        obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    global_step = 0
    num_updates = cfg.total_timesteps // cfg.rollout_steps

    # ---- 可选：记录 cfg/args（外部 init 已写 config 的话可省略）----
    if wandb_run is not None:
        wandb_run.log(
            {
                "train/seed": cfg.seed,
                "train/rollout_steps": cfg.rollout_steps,
                "train/total_timesteps": cfg.total_timesteps,
                "train/num_updates": num_updates,
            },
            step=global_step,
        )

    for update in range(1, num_updates + 1):
        t_update0 = time.time()

        # ---- LR anneal ----
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / float(num_updates)
            lr_now = cfg.lr * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
        else:
            lr_now = optimizer.param_groups[0]["lr"]

        # reset buffer
        buffer.reset(global_step=global_step)

        # -------- Rollout: 收集 cfg.rollout_steps 次仿真 --------
        rollout_rewards = []
        rollout_success = []

        for _ in range(cfg.rollout_steps):
            global_step += 1

            with torch.no_grad():
                action_t, logprob_t, value_t = model.act(obs_t)

            action_np = action_t.detach().cpu().numpy()

            # 让 env 自己把 global_step 用于 wandb 的 step 对齐
            if GYMNASIUM:
                next_obs, reward, terminated, truncated, info = env.step(action_np, global_step=global_step)
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = env.step(action_np, global_step=global_step)

            # reward -> tensor
            reward_t = torch.tensor(reward, device=device, dtype=torch.float32)

            buffer.add(
                obs=obs_t,
                action=action_t,
                logprob=logprob_t,
                reward=reward_t,
                done=done,
                value=value_t,
                global_step=global_step,
            )

            rollout_rewards.append(float(reward))
            if isinstance(info, dict) and "success" in info:
                rollout_success.append(float(bool(info["success"])))

            # one-shot：每步都 reset
            if GYMNASIUM:
                next_obs = env.reset()
            else:
                next_obs = env.reset()
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # bootstrap value (虽然 one-shot，仍保持结构一致)
        with torch.no_grad():
            _, last_value = model.forward(obs_t)

        buffer.compute_gae(
            last_value=last_value,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            global_step=global_step,
        )

        data = buffer.get()
        b_obs = data["obs"]
        b_actions = data["actions"]
        b_logprobs = data["logprobs"]
        b_adv = data["advantages"]
        b_returns = data["returns"]
        b_values = data["values"]

        if cfg.normalize_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        batch_size = cfg.rollout_steps
        minibatch_size = batch_size // cfg.num_minibatches
        inds = np.arange(batch_size)

        approx_kl = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        # ---- PPO update ----
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_adv = b_adv[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_old_values = b_values[mb_inds]

                new_logprob, entropy, new_value = model.evaluate_actions(mb_obs, mb_actions)

                log_ratio = new_logprob - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # 可选：value clip（如果你 cfg 里有）
                if getattr(cfg, "clip_vloss", False):
                    v_clipped = mb_old_values + torch.clamp(new_value - mb_old_values, -cfg.clip_coef, cfg.clip_coef)
                    vloss1 = (new_value - mb_returns).pow(2)
                    vloss2 = (v_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(vloss1, vloss2).mean()
                else:
                    value_loss = 0.5 * (mb_returns - new_value).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # ---- Metrics ----
        with torch.no_grad():
            explained_var = 1.0 - torch.var(b_returns - b_values) / (torch.var(b_returns) + 1e-8)
            mean_return = float(b_returns.mean().item())
            mean_reward = float(np.mean(rollout_rewards)) if rollout_rewards else float("nan")
            success_rate = float(np.mean(rollout_success)) if rollout_success else float("nan")
            update_time = time.time() - t_update0

        # ---- Print ----
        print(
            f"update={update:4d}/{num_updates}  sims={global_step:6d}  "
            f"policy_loss={policy_loss.item(): .4f}  value_loss={value_loss.item(): .4f}  "
            f"entropy={entropy_loss.item(): .4f}  approx_kl={approx_kl.item(): .4f}  "
            f"explained_var={explained_var.item(): .3f}  mean_return={mean_return: .4f}  "
            f"mean_reward={mean_reward: .4f}  success_rate={success_rate: .3f}  "
            f"lr={lr_now: .3e}  time={update_time: .1f}s"
        )

        # ---- Wandb log ----
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/update": update,
                    "train/global_step": global_step,
                    "train/lr": lr_now,
                    "loss/policy": float(policy_loss.item()),
                    "loss/value": float(value_loss.item()),
                    "loss/entropy": float(entropy_loss.item()),
                    "loss/approx_kl": float(approx_kl.item()),
                    "metrics/explained_var": float(explained_var.item()),
                    "metrics/mean_return": mean_return,
                    "metrics/mean_reward": mean_reward,
                    "metrics/success_rate": success_rate,
                    "time/update_sec": update_time,
                },
                step=global_step,
            )
