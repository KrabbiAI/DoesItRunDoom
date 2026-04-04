#!/usr/bin/env python3
"""
Ludicrous Speed - Play/Inference Script
Load trained model and play Doom, optionally record video.
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doom_env import DoomEnv


def play(model_path, scenario="basic", n_episodes=3, render=True, record=False, record_dir="./videos"):
    """Run inference with trained model"""

    if not os.path.exists(model_path):
        print(f"[Ludicrous Speed] Model not found: {model_path}")
        return

    print(f"[Ludicrous Speed] Loading model from {model_path}")
    model = PPO.load(model_path)

    env = DoomEnv(scenario=scenario, visible=render)

    if record:
        os.makedirs(record_dir, exist_ok=True)
        env.game.set_window_visible(True)  # Must be visible to record
        env.game.set_video_recording_enabled(True)

    print(f"[Ludicrous Speed] Playing {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

            if render and step % 10 == 0:
                print(f"  Episode {ep+1} | Step {step} | Reward: {ep_reward:.1f}")

        print(f"[Ludicrous Speed] Episode {ep+1} finished | Reward: {ep_reward:.1f} | Steps: {step}")

        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_path = os.path.join(record_dir, f"ludicrous_ep{ep+1}_{timestamp}.mp4")
            env.game.change_record(recode=True, file_name=rec_path)
            print(f"  Recording saved: {rec_path}")

    env.close()
    print("[Ludicrous Speed] Done playing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ludicrous Speed - Play trained agent")
    parser.add_argument("--model", type=str, default=None, help="Path to model (.zip)")
    parser.add_argument("--scenario", type=str, default="basic", help="Scenario name")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--record", action="store_true", help="Record gameplay to video")
    parser.add_argument("--record-dir", type=str, default="./videos", help="Video output directory")
    args = parser.parse_args()

    # Default model path
    if not args.model:
        default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "doom_ppo_ludicrous.zip")
        if os.path.exists(default):
            args.model = default
        else:
            print("[Ludicrous Speed] No model specified and no default model found!")
            sys.exit(1)

    play(args.model, args.scenario, args.episodes, render=True, record=args.record, record_dir=args.record_dir)
