"""
DoesItRunDoom? — Training Module
Trains PPO agent for 60 minutes then saves and exits cleanly.
"""

import os
import sys
import time
import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))
from notify import TelegramNotifier
from config import SCENARIOS


class TrainingCallback(BaseCallback):
    """Tracks episode count during training."""

    def __init__(self, notifier: TelegramNotifier, outdir: str, verbose: int = 0):
        super().__init__(verbose)
        self.notifier = notifier
        self.outdir = outdir
        self.episode_count = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            self.episode_count += 1
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        self.notifier.send(
            f"✅ Training complete!\n"
            f"🎮 {self.episode_count} episodes\n"
            f"⏱️  {elapsed/60:.1f} min"
        )


def train(
    outdir: str,
    scenario: str,
    duration_min: int = 60,
    total_timesteps: int | None = None,
):
    """Train PPO agent for specified duration (minutes)."""
    notifier = TelegramNotifier()
    notifier.send(f"🚀 Training started!\n📁 {outdir}\n🎮 {scenario}\n⏱️  {duration_min} min")

    scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["deadly_corridor"])
    env_id = scenario_cfg["env_id"]
    env_config = scenario_cfg.get("env_config", {})

    # Create env (no video recording here — handled separately)
    env = gym.make(env_id, **env_config)
    env = Monitor(env, outdir)

    # Hyperparameters from config
    cfg = scenario_cfg.get("ppo", {})
    model = PPO(
        "CnnPolicy",
        env,
        tensorboard_log=os.path.join(outdir, "tensorboard"),
        verbose=1,
        **cfg
    )

    callback = TrainingCallback(notifier, outdir)

    # Calculate timesteps from duration (approx 10 eps/min for deadly_corridor)
    steps_per_minute = 10 * scenario_cfg.get("ep_timeout", 2100)
    if total_timesteps is None:
        total_timesteps = duration_min * steps_per_minute

    print(f"📊 Training for ~{total_timesteps} timesteps ({duration_min} min)")

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    elapsed = time.time() - start

    # Save model
    model_path = os.path.join(outdir, "final_model.zip")
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")

    notifier.send(
        f"✅ Training done!\n"
        f"⏱️  {elapsed/60:.1f} min | {callback.episode_count} episodes\n"
        f"💾 {model_path}"
    )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Doom RL Agent")
    parser.add_argument("--outdir", type=str, default="runs/default", help="Output directory")
    parser.add_argument("--scenario", type=str, default="deadly_corridor")
    parser.add_argument("--duration", type=int, default=60, help="Training duration in minutes")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train(
        outdir=args.outdir,
        scenario=args.scenario,
        duration_min=args.duration,
    )
