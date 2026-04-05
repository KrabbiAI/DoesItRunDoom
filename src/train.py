"""
DoesItRunDoom? — Training Module
Trains PPO agent for 60 minutes then saves and exits cleanly.
"""

import json
import os
import signal
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

# Graceful shutdown handler
_graceful_shutdown = [False]

def _sigterm_handler(signum, frame):
    _graceful_shutdown[0] = True
    print("\n🛑 SIGTERM empfangen — trainiere episode zu Ende...")

signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)
from config import SCENARIOS
from env import ScreenOnlyWrapper


class TrainingCallback(BaseCallback):
    """Tracks episode count during training."""

    def __init__(self, notifier: TelegramNotifier, outdir: str, duration_min: int = 60, env_id: str = "", scenario: str = "", graceful_shutdown: list = None, verbose: int = 0):
        super().__init__(verbose)
        self.notifier = notifier
        self.outdir = outdir
        self.duration_min = duration_min
        self.env_id = env_id
        self.scenario = scenario
        self.episode_count = 0
        self.start_time = time.time()
        self.last_status_time = self.start_time
        self.total_timesteps = 0
        self._last_reported_elapsed = 0.0  # for delta calculation
        self.stats = {"total_training_min": 0}
        self.graceful_shutdown = graceful_shutdown if graceful_shutdown is not None else [False]

    def _on_step(self) -> bool:
        if self.graceful_shutdown and self.graceful_shutdown[0]:
            print("\n🛑 Graceful shutdown — stoppe nach diesem step")
            return False  # Stop training after this step
        if len(self.model.ep_info_buffer) > 0:
            self.episode_count += 1
        self.total_timesteps += 1
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        # Add only the delta from last reported elapsed
        delta_min = (elapsed - self._last_reported_elapsed) / 60
        self.stats['total_training_min'] = self._get_cumulative_min() + delta_min
        self._save_stats(self.stats)
        total_str = self._fmt_duration(int(self.stats['total_training_min']))
        expected_end = datetime.fromtimestamp(self.start_time + self.duration_min * 60).strftime('%H:%M')
        self.notifier.send(
            f"✅ Training complete!\n"
            f"⏱️  {elapsed/60:.1f} min\n"
            f"🎮 {self.episode_count} episodes\n"
            f"📈 Gesamte Trainingszeit: {total_str}"
        )

    def send_status(self) -> None:
        """Send a status update."""
        elapsed = time.time() - self.start_time
        sps = getattr(self, 'steps_per_sec', 0) or 0
        delta_min = (elapsed - self._last_reported_elapsed) / 60
        self.stats['total_training_min'] = self._get_cumulative_min() + delta_min
        self._last_reported_elapsed = elapsed
        self._save_stats(self.stats)
        total_str = self._fmt_duration(int(self.stats['total_training_min']))
        expected_end = datetime.fromtimestamp(self.start_time + self.duration_min * 60).strftime('%H:%M')
        self.notifier.send(
            f"🏋️ Training läuft noch\n"
            f"──────────────\n"
            f"⏱️  {elapsed/60:.1f}/{self.duration_min} min\n"
            f"🕐 Bis {expected_end} Uhr\n"
            f"🎮 {self.episode_count} episodes\n"
            f"📊 {self.total_timesteps} timesteps\n"
            f"⚡ {sps:.0f} steps/s\n"
            f"📈 Gesamte Trainingszeit: {total_str}"
        )

    def _load_stats(self) -> dict:
        """Load ALL stats from file."""
        path = os.path.join(self.outdir, "training_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_stats(self, stats: dict) -> None:
        """Save stats file preserving other scenarios."""
        path = os.path.join(self.outdir, "training_stats.json")
        all_stats = {}
        if os.path.exists(path):
            with open(path) as f:
                all_stats = json.load(f)
        # Update this scenario's data
        all_stats[self.scenario] = {
            'total_training_min': stats.get('total_training_min', 0)
        }
        with open(path, 'w') as f:
            json.dump(all_stats, f, indent=2)

    def _get_cumulative_min(self) -> float:
        """Get cumulative training minutes for THIS scenario."""
        all_stats = self._load_stats()
        return all_stats.get(self.scenario, {}).get('total_training_min', 0)

    def _fmt_duration(self, total_min: int) -> str:
        """Format minutes as 'Xd Xh Xm'."""
        days = total_min // (60 * 24)
        hours = (total_min % (60 * 24)) // 60
        mins = total_min % 60
        parts = []
        if days > 0: parts.append(f"{days}d")
        if hours > 0: parts.append(f"{hours}h")
        parts.append(f"{mins}m")
        return " ".join(parts)


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
    scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["deadly_corridor"])
    env_id = scenario_cfg["env_id"]

    scenario_name = scenario_cfg.get("name", scenario)
    next_status = datetime.fromtimestamp(time.time() + 300).strftime('%H:%M')
    notifier.send(f"🚀 Training gestartet!\n📁 {outdir}\n🎮 {scenario_name}\n⏱️  {duration_min} min\n🕐 {datetime.now().strftime('%H:%M')} → {datetime.fromtimestamp(time.time() + duration_min * 60).strftime('%H:%M')}\n📬 Nächstes Status-Update: {next_status} Uhr")

    env_cls = scenario_cfg["env_cls"]
    env_config = scenario_cfg.get("env_config", {})

    # Create env directly (not via gymnasium.make — envs not registered)
    env = env_cls(**env_config)
    env = ScreenOnlyWrapper(env)
    env = Monitor(env, outdir)

    # Hyperparameters from config
    cfg = scenario_cfg.get("ppo", {})
    model_path = os.path.join(outdir, "final_model")
    if os.path.exists(model_path + ".zip"):
        print(f"📂 Model geladen von: {model_path}")
        model = PPO.load(model_path, env=env)
        # Reset buffer fürsauberes Weitertraining
        model.ep_info_buffer = []
        model.ep_success_buffer = []
    else:
        print(f"🆕 Neues Modell erstellt")
        model = PPO(
            "CnnPolicy",
            env,
            tensorboard_log=os.path.join(outdir, "tensorboard"),
            verbose=1,
            **cfg
        )

    inner_callback = TrainingCallback(notifier, outdir, duration_min, env_id, scenario, _graceful_shutdown)
    callback = StatusCallback(inner_callback)

    # Fixed: 1.7 * ep_timeout gives ~60min training at 58 steps/s
    # (60 * 1.7 * 2100 = 214,200 timesteps / 58 steps/s = ~61 min)
    steps_per_minute = 1.7 * scenario_cfg.get("ep_timeout", 2100)
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
    model_path = os.path.join(outdir, "final_model")
    model.save(model_path)
    print(f"💾 Model saved: {model_path}")

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