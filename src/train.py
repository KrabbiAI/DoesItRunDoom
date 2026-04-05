"""
DoesItRunDoom? — Training Module
Trains PPO agent for 60 minutes then saves and exits cleanly.
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))
from notify import TelegramNotifier
from config import SCENARIOS
from env import ScreenOnlyWrapper


class TrainingCallback(BaseCallback):
    """Tracks episode count during training."""

    def __init__(self, notifier: TelegramNotifier, outdir: str, verbose: int = 0):
        super().__init__(verbose)
        self.notifier = notifier
        self.outdir = outdir
        self.episode_count = 0
        self.start_time = time.time()
        self.last_status_time = self.start_time
        self.total_timesteps = 0

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            self.episode_count += 1
        self.total_timesteps += 1
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        self.notifier.send(
            f"✅ Training complete!\n"
            f"🎮 {self.episode_count} episodes\n"
            f"⏱️  {elapsed/60:.1f} min"
        )

    def send_status(self) -> None:
        """Send a status update."""
        elapsed = time.time() - self.start_time
        sps = getattr(self, 'steps_per_sec', 0) or 0
        remaining = (self.total_timesteps / sps / 60) if sps > 0 else 0
        self.notifier.send(
            f"🏋️ Training läuft noch\n"
            f"⏱️  {elapsed/60:.1f} min vergangen\n"
            f"🎮 {self.episode_count} episodes\n"
            f"📊 {self.total_timesteps} timesteps\n"
            f"⚡ {sps:.0f} steps/s | 🔮 ~{remaining:.0f} min übrig"
        )


class StatusCallback(BaseCallback):
    """Sends a status update every N seconds."""

    def __init__(self, inner_callback: TrainingCallback, status_interval_sec: int = 300, verbose: int = 0):
        super().__init__(verbose)
        self.inner = inner_callback
        self.last_status = time.time()
        self.status_interval = status_interval_sec

    def _on_step(self) -> bool:
        self.inner.total_timesteps += 1
        if len(self.model.ep_info_buffer) > 0:
            self.inner.episode_count += 1
        now = time.time()
        if now - self.last_status >= self.status_interval:
            elapsed = now - self.inner.start_time
            self.inner.steps_per_sec = self.inner.total_timesteps / elapsed if elapsed > 0 else 1
            self.inner.send_status()
            self.last_status = now
        return True


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
    env_cls = scenario_cfg["env_cls"]
    env_config = scenario_cfg.get("env_config", {})

    # Create env directly (not via gymnasium.make — envs not registered)
    env = env_cls(**env_config)
    env = ScreenOnlyWrapper(env)
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

    inner_callback = TrainingCallback(notifier, outdir)
    callback = StatusCallback(inner_callback)

    # Calculate timesteps from duration (approx 10 eps/min for deadly_corridor)
    steps_per_minute = 10 * scenario_cfg.get("ep_timeout", 2100)
    if total_timesteps is None:
        total_timesteps = duration_min * steps_per_minute

    print(f"📊 Training for ~{total_timesteps} timesteps ({duration_min} min)")
    inner_callback.steps_per_sec = 0  # will be calculated during training

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
        f"⏱️  {elapsed/60:.1f} min | {inner_callback.episode_count} episodes\n"
        f"💾 {model_path}"
    )

    env.close()

    # Record video of the trained agent playing
    import subprocess
    notifier.send("🎬 Starte Video-Aufnahme des trainierten Agenten...")
    try:
        result = subprocess.run(
            ["python3", os.path.join(os.path.dirname(__file__), "play.py"), "--model", model_path, "--scenario", scenario],
            capture_output=True, text=True, timeout=300
        )
        notifier.send(f"📹 Video-Aufnahme abgeschlossen!")
    except Exception as e:
        notifier.send(f"⚠️ Video fehlgeschlagen: {e}")


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