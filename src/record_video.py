#!/usr/bin/env python3
"""
Ludicrous Speed - Record 2-minute gameplay video and send to Telegram.
Run after training to capture agent playing Doom.
"""

import os
import sys
import glob
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from doom_env import DoomEnv


def record_video(model_path, output_path, duration_sec=120, scenario="basic"):
    """Record gameplay for specified duration"""

    if not os.path.exists(model_path):
        print(f"[Ludicrous Speed] Model not found: {model_path}")
        return None

    print(f"[Ludicrous Speed] Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create env (visible so VizDoom renders)
    env = DoomEnv(scenario=scenario, visible=True)
    env.game.set_screen_resolution(vizdoom_import().ScreenResolution.RES_640X480)
    env.reset()

    fps = 35
    max_steps = duration_sec * fps
    total_steps = 0
    episode_count = 0

    # Output dir
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Ludicrous Speed] Recording ~{duration_sec}s of gameplay...")

    while total_steps < max_steps:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_path = os.path.join(out_dir, f"ludicrous_ep{episode_count+1}_{timestamp}.mp4")

        # Configure recording for this episode
        env.game.set_video_recording_enabled(True)
        env.game.change_record(recode=True, file_name=rec_path)

        obs, info = env.reset()
        ep_reward = 0
        done = False
        step = 0

        episode_count += 1
        while not done and total_steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            total_steps += 1
            done = terminated or truncated

        print(f"  Episode {episode_count}: reward={ep_reward:.1f}, steps={step}")

        # Episode ended — recording saved automatically
        env.game.set_video_recording_enabled(False)

        if total_steps >= max_steps:
            break

    env.game.close()
    print(f"[Ludicrous Speed] Done. Check videos/ directory.")

    # Return latest video
    rec_files = glob.glob(os.path.join(out_dir, "ludicrous_ep*.mp4"))
    return max(rec_files, key=os.path.getmtime) if rec_files else None


def vizdoom_import():
    import vizdoom
    return vizdoom


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="basic")
    args = parser.parse()

    # Find latest model if not specified
    if not args.model:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        models = glob.glob(os.path.join(models_dir, "*.zip"))
        if not models:
            print("[Ludicrous Speed] No model found in models/")
            sys.exit(1)
        args.model = max(models)
        print(f"[Ludicrous Speed] Using latest model: {args.model}")

    video = record_video(args.model, args.output, args.duration, args.scenario)
    if video:
        print(f"[Ludicrous Speed] Video saved: {video}")
    else:
        print("[Ludicrous Speed] Recording failed")
