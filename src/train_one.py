#!/usr/bin/env python3
"""
DoesItRunDoom? - Train for ONE episode and stop.
Episode-based training: run one episode → notify → exit.
Use doom_continue to keep training.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import DoomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from datetime import datetime
import torch

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class EpisodeCallback(BaseCallback):
    """Stop after first episode completion"""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_done = False

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                elapsed = (datetime.now() - self.start_time).total_seconds()
                print(f"[DoesItRunDoom] Episode done! Reward: {ep['r']:.1f} | Steps: {ep['l']} | Time: {elapsed:.0f}s")
                self.episode_done = True
                return False  # Stop the training
        return True

    def _on_rollout_end(self) -> bool:
        if not self.episode_done:
            # Force stop if rollout ended without episode
            print("[DoesItRunDoom] Rollout ended, forcing stop")
            return False
        return True


def train_one_episode():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "doom_ppo_latest.zip")

    # Load existing model or create new
    env = DoomEnv(scenario="deadly_corridor", visible=False)

    if os.path.exists(model_path):
        print(f"[DoesItRunDoom] Loading existing model: {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("[DoesItRunDoom] Creating new model")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
            policy_kwargs={"net_arch": [256, 256]},
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print("[DoesItRunDoom] Starting 1 episode...")

    cb = EpisodeCallback()
    cb.start_time = datetime.now()

    try:
        model.learn(1_000_000, callback=cb, reset_num_timesteps=False)
    except Exception as e:
        print(f"[DoesItRunDoom] Training stopped: {e}")

    # Save model
    model.save(model_path)
    print(f"[DoesItRunDoom] Model saved: {model_path}")

    env.close()
    return cb.episode_done


if __name__ == "__main__":
    done = train_one_episode()
    print(f"[DoesItRunDoom] Done. Episode completed: {done}")
    sys.exit(0 if done else 1)
