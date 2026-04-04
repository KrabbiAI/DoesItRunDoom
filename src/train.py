#!/usr/bin/env python3
"""
Ludicrous Speed - Training Script
VizDoom RL Agent with Stable Baselines3 PPO + TensorBoard
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList, BaseCallback, EvalCallback, TensorboardCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doom_env import DoomEnv


class ProgressCallback(BaseCallback):
    """Prints training progress every N episodes"""

    def __init__(self, print_freq=1000, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.ep_count = 0
        self.start_time = datetime.now()

    def _on_step(self) -> bool:
        # Check for episode terminations via infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_count += 1
                if self.ep_count % self.print_freq == 0:
                    ep = info["episode"]
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    reward = ep["r"]
                    steps = ep["l"]
                    print(f"[Ludicrous Speed] Episode {self.ep_count} | "
                          f"Reward: {reward:.1f} | Steps: {steps} | "
                          f"Time: {elapsed:.0f}s | FPS: {self.num_timesteps / elapsed:.0f}")
        return True

    def _on_rollout_end(self) -> None:
        pass


def make_env(scenario="basic", visible=False, rank=0):
    """Create a doom environment, wrapped with Monitor"""
    def _init():
        env = DoomEnv(scenario=scenario, visible=visible)
        env = Monitor(env, filename=None)  # monitor logging handled by SB3
        return env
    return _init


def train(
    scenario="basic",
    total_timesteps=500_000,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    visible=False,
    model_path=None,
    log_dir="./logs"
):
    print(f"[Ludicrous Speed] Starting training | Scenario: {scenario} | Visible: {visible}")
    print(f"[Ludicrous Speed] Total timesteps: {total_timesteps:,} | LR: {learning_rate}")

    os.makedirs(log_dir, exist_ok=True)

    # Create VecEnv
    env = DummyVecEnv([make_env(scenario=scenario, visible=visible, rank=0)])

    # Tensorboard log dir
    tb_log = os.path.join(log_dir, "tensorboard")

    # Callbacks
    progress = ProgressCallback(print_freq=500)
    tensorboard_cb = TensorboardCallback(tb_log, verbose=0)

    callbacks = CallbackList([progress, tensorboard_cb])

    # Load existing model or create new
    if model_path and os.path.exists(model_path):
        print(f"[Ludicrous Speed] Loading existing model: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=tb_log)
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        print("[Ludicrous Speed] Creating new PPO model")
        policy_kwargs = {
            "net_arch": [256, 256],  # Medium CNN policy
        }

        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=0,
            tensorboard_log=tb_log,
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            callback=callbacks,
            tb_log_name=datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    # Save model
    models_dir = os.path.join(os.path.dirname(log_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "doom_ppo_ludicrous")
    model.save(save_path)
    print(f"[Ludicrous Speed] Model saved to {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ludicrous Speed - Doom RL Trainer")
    parser.add_argument("--scenario", type=str, default="basic", help="VizDoom scenario name")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--visible", action="store_true", help="Show VizDoom window during training")
    parser.add_argument("--model", type=str, default=None, help="Path to existing model to continue training")
    parser.add_argument("--logdir", type=str, default="./logs", help="Log directory")
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        visible=args.visible,
        model_path=args.model,
        log_dir=args.logdir,
    )
