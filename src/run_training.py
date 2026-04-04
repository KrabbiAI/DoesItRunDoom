#!/usr/bin/env python3
"""
Ludicrous Speed - TensorBoard + Training Runner
Starts both TensorBoard server and the training process.
Auto-stops after duration (seconds) and sends Telegram notification.
"""

import subprocess
import sys
import os
import signal
import time
import argparse
import threading
from datetime import datetime


def start_tensorboard(logdir, port=6006):
    """Start TensorBoard in background"""
    print(f"[Ludicrous Speed] Starting TensorBoard on http://localhost:{port}")
    tb_proc = subprocess.Popen(
        ["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return tb_proc


def send_telegram(message):
    """Send notification via OpenClaw bot"""
    try:
        import requests
        import json
        creds_path = os.path.expanduser("~/.openclaw/workspace/credentials.json")
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                creds = json.load(f)
                bot_token = creds.get("telegram_bot_token")
                chat_id = creds.get("telegram_chat_id")
            if bot_token and chat_id:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as e:
        print(f"[Ludicrous Speed] Telegram notification failed: {e}")


def start_training(scenario, timesteps, lr, visible, model_path, logdir):
    """Start the training process"""
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--scenario", scenario,
        "--timesteps", str(timesteps),
        "--lr", str(lr),
        "--logdir", logdir,
    ]
    if visible:
        cmd.append("--visible")
    if model_path:
        cmd.extend(["--model", model_path])

    print(f"[Ludicrous Speed] Starting training: {' '.join(cmd)}")
    train_proc = subprocess.Popen(cmd)
    return train_proc


def watchdog(train_proc, duration, tb_proc):
    """Auto-stop after duration seconds"""
    print(f"[Ludicrous Speed] Watchdog started: will stop in {duration}s")
    start = time.time()
    while time.time() - start < duration:
        if train_proc.poll() is not None:
            # Training already ended
            break
        time.sleep(5)

    elapsed = int(time.time() - start)
    print(f"[Ludicrous Speed] Duration reached ({elapsed}s). Stopping training...")

    # Stop training
    train_proc.terminate()
    try:
        train_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        train_proc.kill()

    # Stop TensorBoard
    tb_proc.terminate()
    tb_proc.wait(timeout=5)

    # Get server IP
    server_ip = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True
    ).stdout.strip().split()[0]

    msg = (
        f"🏎️ Ludicrous Speed — Training fertig!\n"
        f"⏱️ Dauer: {elapsed}s\n"
        f"📊 TensorBoard: http://{server_ip}:6006\n"
        f"\n"
        f"Nach dem Training ein Video aufnehmen?\n"
        f"/doom_video"
    )
    send_telegram(msg)
    print("[Ludicrous Speed] Done. Telegram notification sent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="basic")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--duration", type=int, default=600, help="Training duration in seconds")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    tb_proc = start_tensorboard(args.logdir, args.tensorboard_port)
    train_proc = start_training(
        args.scenario, args.timesteps, args.lr,
        args.visible, args.model, args.logdir
    )

    print(f"[Ludicrous Speed] Training PID: {train_proc.pid}")
    print(f"[Ludicrous Speed] TensorBoard PID: {tb_proc.pid}")

    # Get server IP for display
    server_ip = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True
    ).stdout.strip().split()[0]
    print(f"[Ludicrous Speed] TensorBoard: http://{server_ip}:{args.tensorboard_port}")
    print(f"[Ludicrous Speed] Training for {args.duration}s (~{args.duration//60} min)")

    # Start watchdog thread
    watchdog_thread = threading.Thread(
        target=watchdog,
        args=(train_proc, args.duration, tb_proc),
        daemon=True
    )
    watchdog_thread.start()

    try:
        train_proc.wait()
    except KeyboardInterrupt:
        print("\n[Ludicrous Speed] Interrupted. Stopping...")
        train_proc.terminate()
        tb_proc.terminate()
