#!/usr/bin/env python3
"""
DoesItRunDoom? - Runner
Starts training for ONE episode, notifies when done.
Run via: ./ludicrous.sh start
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "logs")


def main():
    print("[DoesItRunDoom] Starting 1 episode...")

    # Start tensorboard
    tb_proc = subprocess.Popen(
        ["tensorboard", "--logdir", LOG_DIR, "--port", "6006", "--bind_all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    start_time = time.time()

    try:
        # Run training for 1 episode
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "train_one.py")],
            timeout=600  # 10 min max per episode
        )
    finally:
        tb_proc.terminate()
        tb_proc.wait(timeout=5)

    elapsed = int(time.time() - start_time)

    # Get server IP
    server_ip = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True
    ).stdout.strip().split()[0]

    # Launch notification
    notify_script = os.path.join(SCRIPT_DIR, "notify_done.py")
    subprocess.Popen(
        [sys.executable, notify_script, str(elapsed)],
        stdout=open(os.path.join(LOG_DIR, "notify.log"), "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    print("[DoesItRunDoom] Done.")


if __name__ == "__main__":
    main()
