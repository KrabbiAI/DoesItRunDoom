#!/usr/bin/env python3
"""
DoesItRunDoom? - Headless video recorder
Captures Doom screen buffers and encodes to MP4 video — no display needed!
Uses OpenCV to encode RGB frames to H.264.
"""

import os
import sys
import glob
import time
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from doom_env import DoomEnv

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[Record] OpenCV not available, falling back to VizDoom recording")


def record_video_headless(model_path, output_path, duration_sec=120, scenario="basic"):
    """Record Doom gameplay using screen buffers + OpenCV encoding"""

    if not os.path.exists(model_path):
        print(f"[Record] Model not found: {model_path}")
        return None

    if not HAS_CV2:
        print("[Record] OpenCV not available, cannot record headless")
        return None

    print(f"[Record] Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create env in headless mode
    env = DoomEnv(scenario=scenario, visible=False)
    obs, info = env.reset()
    display_w, display_h = 640, 480

    # Use training resolution (160x120) for inference, upscale for video
    fps = 30
    display_w, display_h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (display_w, display_h))

    print(f"[Record] Recording {duration_sec}s headless to {output_path}")

    fps_real = 35
    max_steps = duration_sec * fps_real
    total_steps = 0
    episode_count = 0
    start_time = time.time()

    try:
        while total_steps < max_steps:
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

                # Capture frame from screen buffer
                state = env.game.get_state()
                if state is not None:
                    frame = state.screen_buffer
                    if frame is not None:
                        # Resize 160x120 to 640x480 for video
                        frame_resized = cv2.resize(frame, (display_w, display_h))
                        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)

            elapsed = time.time() - start_time
            print(f"  Episode {episode_count}: reward={ep_reward:.1f}, steps={step}, elapsed={elapsed:.1f}s")

    finally:
        out.release()
        env.close()

    actual_duration = time.time() - start_time
    print(f"[Record] Done! {episode_count} episodes, {actual_duration:.1f}s, saved to {output_path}")

    # Convert to proper MP4 with ffmpeg
    try:
        mp4_path = output_path.replace('.mp4', '_converted.mp4')
        subprocess.run([
            'ffmpeg', '-y', '-i', output_path,
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', mp4_path
        ], capture_output=True, timeout=60)
        os.replace(mp4_path, output_path)
        print(f"[Record] Converted to MP4: {output_path}")
    except Exception as e:
        print(f"[Record] FFmpeg conversion failed: {e}")

    return output_path


def record_video_fallback(model_path, output_path, duration_sec=120, scenario="basic"):
    """Fallback: use VizDoom's built-in video recording"""
    if not os.path.exists(model_path):
        print(f"[Record] Model not found: {model_path}")
        return None

    print(f"[Record] Loading model: {model_path}")
    model = PPO.load(model_path)

    env = DoomEnv(scenario=scenario, visible=True)
    env.reset()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec_path = os.path.join(os.path.dirname(output_path), f"ludicrous_{timestamp}.mp4")

    env.game.set_video_recording_enabled(True)
    env.game.change_record(recode=True, file_name=rec_path)

    fps = 35
    max_steps = duration_sec * fps
    total_steps = 0

    print(f"[Record] Recording to {rec_path}")

    while total_steps < max_steps:
        obs, info = env.reset()
        done = False
        while not done and total_steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            done = terminated or truncated

    env.game.set_video_recording_enabled(False)
    env.close()
    print(f"[Record] Done: {rec_path}")
    return rec_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="basic")
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_files = glob.glob(os.path.join(models_dir, "doom_ppo_*.zip"))
    if not model_files:
        print("[Record] No model found!")
        sys.exit(1)
    model_path = max(model_files)
    print(f"[Record] Using: {model_path}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or os.path.join(out_dir, f"ludicrous_{timestamp}.mp4")

    if HAS_CV2:
        video = record_video_headless(model_path, output, args.duration, args.scenario)
    else:
        video = record_video_fallback(model_path, output, args.duration, args.scenario)

    if video:
        print(f"[Record] Success: {video}")
    else:
        print("[Record] Failed")
        sys.exit(1)
