import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- gym / gymnasium 兼容导入 ---
try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym
    GYMNASIUM = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    total_timesteps: int = 200_000
    rollout_steps: int = 2048          # 每次收集多少步 on-policy 数据
    num_minibatches: int = 32          # rollout_steps * num_envs / num_minibatches = minibatch_size
    update_epochs: int = 10            # 每次 rollout 后训练轮数

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    max_grad_norm: float = 0.5

    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    target_kl: float = 0.03           # 可选：KL 过大提前停止

    normalize_adv: bool = True
    anneal_lr: bool = True

class ActorCriticContinuous(nn.Module):
    """
    连续动作：Gaussian policy
    - 输出 mean（维度=act_dim）
    - log_std 通常作为可学习参数（全局），也可以做成网络输出
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)

        # log_std：每个动作维度一个，可学习
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # 可选初始化（稳定一些）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        return mu, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        mu, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)

        action = dist.sample()
        # 多维动作 log_prob 要 sum
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mu, value = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)

        logprob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logprob, entropy, value


class RolloutBuffer:
    """
    存 on-policy 采样数据，并在 rollout 结束后计算 GAE & returns
    """
    def __init__(self, rollout_steps: int, obs_dim: int, device: str):
        self.rollout_steps = rollout_steps
        self.device = device

        self.obs = torch.zeros((rollout_steps, obs_dim), device=device)
        self.actions = torch.zeros((rollout_steps,), device=device, dtype=torch.long)
        self.logprobs = torch.zeros((rollout_steps,), device=device)
        self.rewards = torch.zeros((rollout_steps,), device=device)
        self.dones = torch.zeros((rollout_steps,), device=device)  # done at step t
        self.values = torch.zeros((rollout_steps,), device=device)

        self.advantages = torch.zeros((rollout_steps,), device=device)
        self.returns = torch.zeros((rollout_steps,), device=device)

        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer overflow")

        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """
        GAE:
        delta_t = r_t + gamma*(1-done_t)*V(s_{t+1}) - V(s_t)
        A_t = delta_t + gamma*lambda*(1-done_t)*A_{t+1}
        return_t = A_t + V(s_t)
        """
        adv = 0.0
        for t in reversed(range(self.rollout_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.rollout_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_non_terminal * next_value - self.values[t]
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def get(self) -> Dict[str, torch.Tensor]:
        assert self.ptr == self.rollout_steps, "Buffer not full"
        return {
            "obs": self.obs,
            "actions": self.actions,
            "logprobs": self.logprobs,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values,
        }


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    # gymnasium reset: obs, info；gym reset: obs
    if GYMNASIUM:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    else:
        env.seed(seed)
    return env


def ppo_train(cfg: PPOConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = make_env(cfg.env_id, cfg.seed)
    obs_space = env.observation_space
    act_space = env.action_space

    assert len(obs_space.shape) == 1, "This implementation assumes 1D observation vector."
    assert hasattr(act_space, "n"), "This implementation is for discrete action spaces."

    obs_dim = obs_space.shape[0]
    act_dim = act_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # 初始 obs
    if GYMNASIUM:
        obs, _info = env.reset(seed=cfg.seed)
    else:
        obs = env.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    buffer = RolloutBuffer(cfg.rollout_steps, obs_dim, cfg.device)

    global_step = 0
    num_updates = cfg.total_timesteps // cfg.rollout_steps

    for update in range(1, num_updates + 1):
        # 学习率退火（可选）
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = cfg.lr * frac
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        buffer.ptr = 0

        # -------- Rollout / Collect on-policy data --------
        for _ in range(cfg.rollout_steps):
            global_step += 1
            with torch.no_grad():
                action, logprob, value = model.act(obs_t)

            # 与环境交互
            if GYMNASIUM:
                next_obs, reward, terminated, truncated, _info = env.step(action.item())
                done = terminated or truncated
            else:
                next_obs, reward, done, _info = env.step(action.item())

            buffer.add(
                obs=obs_t,
                action=action,
                logprob=logprob,
                reward=float(reward),
                done=bool(done),
                value=value,
            )

            # 更新 obs
            if done:
                if GYMNASIUM:
                    next_obs, _info = env.reset()
                else:
                    next_obs = env.reset()
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # rollout 结束，bootstrap last_value
        with torch.no_grad():
            _logits, last_value = model.forward(obs_t)
        buffer.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)

        data = buffer.get()
        b_obs = data["obs"]
        b_actions = data["actions"]
        b_logprobs = data["logprobs"]
        b_adv = data["advantages"]
        b_returns = data["returns"]
        b_values = data["values"]

        # 优势归一化（常见稳定技巧）
        if cfg.normalize_adv:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # -------- PPO Update --------
        batch_size = cfg.rollout_steps
        minibatch_size = batch_size // cfg.num_minibatches
        assert minibatch_size > 0

        inds = np.arange(batch_size)

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

                # ratio = pi(a|s) / pi_old(a|s) = exp(new_logp - old_logp)
                log_ratio = new_logprob - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                # Policy loss: clipped surrogate
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Value loss: 常见做法是 (V - return)^2，也可做 value clipping
                value_loss = 0.5 * (mb_returns - new_value).pow(2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                # 可选：KL 过大提前停止（近似）
                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean()
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        # 简单日志（可按需替换为 tensorboard）
        if update % 10 == 0:
            explained_var = 1 - torch.var(b_returns - b_values) / (torch.var(b_returns) + 1e-8)
            print(
                f"update={update:4d}/{num_updates}  steps={global_step:7d}  "
                f"policy_loss={policy_loss.item(): .4f}  value_loss={value_loss.item(): .4f}  "
                f"entropy={entropy_loss.item(): .4f}  approx_kl={approx_kl.item(): .4f}  "
                f"explained_var={explained_var.item(): .3f}"
            )

    env.close()


if __name__ == "__main__":
    cfg = PPOConfig()
    ppo_train(cfg)
