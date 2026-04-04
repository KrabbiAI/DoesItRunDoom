#!/usr/bin/env python3
"""
Ludicrous Speed - Status Checker
Returns current training status and model info.
"""

import os
import sys
import glob
import subprocess
from datetime import datetime


def get_status():
    pidfile = os.path.join(os.path.dirname(__file__), "..", "ludicrous_speed.pid")
    logdir = os.path.join(os.path.dirname(__file__), "..", "logs")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    # Check if running
    running = False
    pid = None
    if os.path.exists(pidfile):
        with open(pidfile) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            running = True
        except OSError:
            running = False
            # Stale PID file
            os.remove(pidfile)

    # Check latest model
    model_files = glob.glob(os.path.join(models_dir, "doom_ppo_ludicrous.zip"))
    latest_model = max(model_files) if model_files else None
    model_mtime = datetime.fromtimestamp(os.path.getmtime(latest_model)).strftime("%Y-%m-%d %H:%M") if latest_model else None

    # Check latest tensorboard log
    tb_runs = os.path.join(logdir, "tensorboard")
    latest_tb = None
    if os.path.exists(tb_runs):
        runs = glob.glob(os.path.join(tb_runs, "*"))
        latest_tb = max(runs) if runs else None

    return {
        "running": running,
        "pid": pid,
        "latest_model": latest_model,
        "model_mtime": model_mtime,
        "latest_tb_run": os.path.basename(latest_tb) if latest_tb else None,
        "logdir": logdir,
    }


def print_status():
    s = get_status()

    print("🏎️  Ludicrous Speed — Status")
    print("─" * 40)

    if s["running"]:
        print(f"✅ TRAINING RUNNING")
        print(f"   PID: {s['pid']}")
    else:
        print(f"⏸️  NOT RUNNING")

    print()
    if s["latest_model"]:
        print(f"📦 Latest Model: {os.path.basename(s['latest_model'])}")
        print(f"   Saved: {s['model_mtime']}")
    else:
        print(f"📦 No model trained yet")

    print()
    print(f"📊 TensorBoard: http://localhost:6006")
    print(f"   → tensorboard --logdir {s['logdir']}")

    print()
    print(f"🎮 Commands:")
    print(f"   /doom_start   — Start training + TensorBoard")
    print(f"   /doom_stop    — Stop training")
    print(f"   /doom_status  — This status")
    print(f"   /doom_video   — Record 2min gameplay and send")


if __name__ == "__main__":
    print_status()
