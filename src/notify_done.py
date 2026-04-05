#!/usr/bin/env python3
"""
DoesItRunDoom? - Notification + Video Sender
Runs after one episode completes.
"""

import os
import sys
import glob
import json
import re
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VIDEOS_DIR = os.path.join(PROJECT_DIR, "videos")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")


def get_telegram_creds():
    try:
        with open("/home/dobby/.openclaw/openclaw.json") as f:
            d = json.load(f)
        bot_token = d.get("channels", {}).get("telegram", {}).get("botToken")
        return bot_token, "631196199"
    except:
        return None, None


def send_telegram(msg):
    bot_token, chat_id = get_telegram_creds()
    if not bot_token:
        print("[Notify] No Telegram bot token found")
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": msg}, timeout=10)
        print(f"[Notify] Sent: {msg[:50]}... → {resp.status_code}")
        return resp.ok
    except Exception as e:
        print(f"[Notify] Telegram failed: {e}")
        return False


def send_video(video_path, caption):
    bot_token, chat_id = get_telegram_creds()
    if not bot_token:
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendVideo"
        with open(video_path, "rb") as f:
            files = {"video": f}
            data = {"chat_id": chat_id, "caption": caption}
            resp = requests.post(url, data=data, files=files, timeout=180)
        return resp.ok
    except Exception as e:
        print(f"[Notify] Video failed: {e}")
        return False


def main():
    elapsed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    server_ip = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True
    ).stdout.strip().split()[0]

    model_path = os.path.join(MODELS_DIR, "doom_ppo_latest.zip")

    # Parse training log for episode stats
    ep_reward = 0
    ep_steps = 0
    try:
        with open(os.path.join(LOG_DIR, "training.log")) as f:
            for line in f:
                if "Episode done" in line:
                    m = re.search(r"Reward: ([-\d.]+)", line)
                    s = re.search(r"Steps: (\d+)", line)
                    if m:
                        ep_reward = float(m.group(1))
                    if s:
                        ep_steps = int(s.group(1))
    except:
        pass

    send_telegram(
        f"🏎️ Episode fertig!\n"
        f"⏱️ Zeit: {elapsed}s\n"
        f"📈 Reward: {ep_reward:.1f} | Steps: {ep_steps}\n"
        f"📊 TensorBoard: http://{server_ip}:6006"
    )

    # Try video recording
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "record_video.py")],
            capture_output=True,
            text=True,
            timeout=180
        )
        videos = glob.glob(os.path.join(VIDEOS_DIR, "ludicrous_*.mp4"))
        if videos:
            latest = max(videos, key=os.path.getmtime)
            ok = send_video(latest, f"🎬 DoesItRunDoom? — Episode mit Reward {ep_reward:.1f}")
            if ok:
                send_telegram("✅ Video geschickt!")
    except:
        pass

    send_telegram(
        f"\n"
        f"/doom_continue — Nächste Episode\n"
        f"/doom_status — Status\n"
        f"/doom_video — Video aufnehmen"
    )
    print("[Notify] Done.")


if __name__ == "__main__":
    main()
