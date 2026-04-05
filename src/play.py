#!/usr/bin/env python3
"""
DoesItRunDoom? — Video Recording Script
Loads the trained model, lets it play one episode until death, records a video, sends to Telegram.
"""

import os
import sys
import time
import argparse
import tempfile
import requests

sys.path.insert(0, os.path.dirname(__file__))
from notify import TelegramNotifier
from config import SCENARIOS
from env import ScreenOnlyWrapper

import cv2
import numpy as np
import vizdoom
from stable_baselines3 import PPO


def record_episode(model_path: str, scenario: str = "deadly_corridor", out_video: str = "/tmp/doom_playthrough.mp4"):
    """Play one episode with the trained model and record a video."""

    notifier = TelegramNotifier()
    scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["deadly_corridor"])
    env_id = scenario_cfg["env_id"]
    notifier.send(f"🎬 Video Recording gestartet — {scenario_cfg.get('name', scenario)} bis zum Tod...")
    env_cls = scenario_cfg["env_cls"]
    env_config = scenario_cfg.get("env_config", {})

    # Create env
    env = env_cls(**env_config)
    env = ScreenOnlyWrapper(env)

    # Load model
    model = PPO.load(model_path)

    # Video recording setup
    fps = 30
    width, height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    # Reset and play until death
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    start_time = time.time()

    notifier.send("🤖 Agent ist gestartet. Bitte warten...")

    max_recording_sec = 180  # 3 minutes timeout
    while not done:
        # Timeout check — stop recording after 3 min
        if time.time() - start_time >= max_recording_sec:
            print(f"⏱️  Recording timeout after {max_recording_sec}s")
            break
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Get raw screen buffer for video
        try:
            raw_state = env.unwrapped.game.get_state()
            if raw_state and raw_state.screen_buffer is not None:
                frame = raw_state.screen_buffer
                # BGR → RGB → BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Boost dark frames (raw Doom render is very dark)
                frame = np.clip(frame * 2.5, 0, 255).astype(np.uint8)
                # Resize if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
        except Exception as e:
            print(f"Frame capture error: {e}")
            # Write a black frame as placeholder
            black = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(black)

        # Progress update every 500 steps
        if steps % 500 == 0:
            elapsed = time.time() - start_time
            notifier.send(f"📊 {steps} steps | reward: {total_reward:.1f} | {elapsed:.0f}s")

    env.close()
    out.release()

    elapsed = time.time() - start_time
    notifier.send(
        f"💀 Agent gestorben!\n"
        f"⏱️  {elapsed:.0f}s | 🎮 {steps} steps | 🏆 {total_reward:.1f} reward\n"
        f"📁 {out_video}"
    )

    return out_video, total_reward, steps, env_id


def send_video(video_path: str, caption: str = "🎮 Doom Agent in Action!"):
    """Send video file via Telegram."""
    bot_token = "8798400513:AAHVGh4T2dtsEXZML6zmtXLNLVPM4lpAcZE"
    chat_id = "631196199"
    url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {'chat_id': chat_id, 'caption': caption}
        r = requests.post(url, data=data, files=files, timeout=120)
        print(f"Telegram response: {r.status_code} {r.text[:200]}")
        return r.status_code == 200



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Doom agent playing video")
    parser.add_argument("--model", type=str, default=None, help="Path to model (auto-selected by scenario if not provided)")
    parser.add_argument("--scenario", type=str, default="deadly_corridor")
    parser.add_argument("--output", type=str, default="/tmp/doom_playthrough.mp4")
    args = parser.parse_args()

    # Auto-select model based on scenario
    if args.model is None:
        scenario_dir = f"runs/{args.scenario}"
        args.model = f"{scenario_dir}/final_model"

    video_path, reward, steps, _env_id = record_episode(args.model, args.scenario, args.output)

    # Convert to Telegram-friendly format (re-encode as h264, heavily compressed)
    temp_mp4 = "/tmp/doom_telegram.mp4"
    os.system(f"/home/dobby/.openclaw/workspace/youtube-shorts/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg -y -i {video_path} -c:v libx264 -crf 42 -preset ultrafast -vf 'scale=240:-2' -maxrate 80k -bufsize 160k -pix_fmt yuv420p -r 15 -g 30 {temp_mp4} > /dev/null 2>&1")

    send_path = temp_mp4 if (os.path.exists(temp_mp4) and os.path.getsize(temp_mp4) > 0) else video_path
    scenario_name = SCENARIOS.get(args.scenario, SCENARIOS["deadly_corridor"]).get("name", args.scenario)
    send_video(send_path, f"🎬 {scenario_name}  📊 {steps} steps | reward {reward:.1f}")
