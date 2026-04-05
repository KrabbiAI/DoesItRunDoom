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
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_DIR, "logs")


def main():
    print("[DoesItRunDoom] Starting 1 episode...")

    # Clean old logs
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(os.path.join(LOG_DIR, "training.log"), "w")

    # Start tensorboard
    tb_proc = subprocess.Popen(
        ["tensorboard", "--logdir", LOG_DIR, "--port", "6006", "--bind_all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    start_time = time.time()

    # Run training for 1 episode
    train_proc = subprocess.Popen(
        [sys.executable, os.path.join(SCRIPT_DIR, "train_one.py")],
        stdout=log_file,
        stderr=subprocess.STDOUT
    )

    # Wait for training to finish
    train_proc.wait()
    log_file.close()

    elapsed = int(time.time() - start_time)

    # Stop TensorBoard
    tb_proc.terminate()
    tb_proc.wait(timeout=5)

    # Get server IP
    server_ip = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True
    ).stdout.strip().split()[0]

    # Launch notification (independent process)
    notify_script = os.path.join(SCRIPT_DIR, "notify_done.py")
    subprocess.Popen(
        [sys.executable, notify_script, str(elapsed)],
        stdout=open(os.path.join(LOG_DIR, "notify.log"), "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    print("[DoesItRunDoom] Episode complete.")


if __name__ == "__main__":
    main()
