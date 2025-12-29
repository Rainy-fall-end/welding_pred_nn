import sys
import os
import wandb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "abaqus_data"))

from rl.dpo import build_batch_args,PPOConfig,ppo_train

if __name__ == "__main__":
    # 1) FEM pipeline 参数（复用 batch_runner 风格）
    args = build_batch_args()

    # 2) PPO 训练
    cfg = PPOConfig(
        total_timesteps=40,   # 例如 200 次仿真
        rollout_steps=4,      # 每次更新收集 10 次仿真
        num_minibatches=4,
        update_epochs=4,
        t_max=100
    )
    # run = wandb.init(
    #     project="Welding_rl",
    # )
    run = None
    ppo_train(cfg, args,wandb_run=run)