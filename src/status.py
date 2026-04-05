#!/usr/bin/env python3
"""DoesItRunDoom? - Status Checker"""

import os
import glob

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(PROJECT_DIR, "logs")
MODEL = os.path.join(PROJECT_DIR, "models", "doom_ppo_latest.zip")

def main():
    print("🏎️  DoesItRunDoom? — Status")
    print("─" * 40)

    # Check if running
    pidfile = os.path.join(PROJECT_DIR, "ludicrous_speed.pid")
    running = False
    if os.path.exists(pidfile):
        with open(pidfile) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            os.remove(pidfile)

    if running:
        print(f"✅ TRAINING LAEUFT (PID {pid})")
    else:
        print(f"⏸️  NICHT LAUFEND")

    # Check model
    if os.path.exists(MODEL):
        import datetime
        mtime = os.path.getmtime(MODEL)
        age = datetime.datetime.fromtimestamp(mtime)
        print(f"📦 Model: doom_ppo_latest.zip")
        print(f"   Gespeichert: {age.strftime('%H:%M:%S')}")
    else:
        print(f"📦 Kein Model vorhanden")

    # Check latest episode log
    train_log = os.path.join(LOGDIR, "training.log")
    if os.path.exists(train_log):
        with open(train_log) as f:
            content = f.read()
        for line in content.split('\n'):
            if 'Episode done' in line:
                print(f"   Letzte Episode: {line.strip()}")
                break

    print()
    print(f"📊 TensorBoard: http://localhost:6006")
    print()
    print(f"📁 Project: {PROJECT_DIR}")
    print()
    print(f"Commands:")
    print(f"   /doom_start    — 1 Episode trainieren")
    print(f"   /doom_continue — Naechste Episode")
    print(f"   /doom_stop     — Stoppen")
    print(f"   /doom_status   — Diesen Status")


if __name__ == "__main__":
    main()
