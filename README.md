# DoesItRunDoom?

**Trains a PPO agent to play Doom using VizDoom + Stable-Baselines3.**

## What This Does

Reinforcement Learning training pipeline that teaches an agent to navigate Doom's Deadly Corridor. Sends Telegram status notifications every 5 minutes, records a video at the end, and tracks cumulative training time across all runs.

## Quick Start

```bash
cd /home/dobby/ludicrous-speed

# Install Python dependencies
pip install vizdoom gymnasium stable-baselines3 numpy requests opencv-python

# VizDoom needs these system libs (Ubuntu/Debian):
sudo apt install cmake libboost-all-dev libboost-python-dev libboost-system-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev libexpat1-dev
sudo apt install zlib1g-dev libsdl2-dev libssl-dev

# Get Doom IWAD files and place them in _vizdoom/
#
# Option A — Doom 2 (needed for deadly_corridor):
wget https://archive.org/download/DOOM2IWADFILE/DOOM2.WAD -O _vizdoom/doom2.wad
#
# Option B — Doom 1 shareware (limited, but free):
wget https://archive.org/download/doom-wads/Doom%20%28v1.9%29.zip -O _vizdoom/doom1.zip
unzip _vizdoom/doom1.zip -d _vizdoom/ && mv _vizdoom/doom/doom.wad _vizdoom/doom1.wad
#
# deadly_corridor needs doom2.wad (Option A)

# Start training with Telegram notifications
cd /home/dobby/ludicrous-speed
TELEGRAM_BOT_TOKEN="<your_bot_token>" TELEGRAM_CHAT_ID="<your_chat_id>" \
  python -m src.train \
    --outdir runs/run_$(date +%Y_%m_%d_%H%M) \
    --duration 60 \
    --scenario deadly_corridor
```

## Telegram Notifications

The training sends Telegram notifications on:
- **Start**: Run info + previous cumulative stats
- **Every 5 min**: Progress update with cumulative training time
- **End**: Final stats + video recording starts
- **Video**: Recording result (checks if file actually exists)

Requires env vars or defaults in `src/notify.py`:
```bash
export TELEGRAM_BOT_TOKEN="<your_bot_token>"
export TELEGRAM_CHAT_ID="<your_chat_id>"
```

## Cumulative Stats

Stats are saved in `runs/<scenario>/stats.json` and persist across all runs:
- `total_training_min` — total training minutes
- `total_episodes` — total episodes
- `total_timesteps` — total timesteps
- `best_reward` — best reward seen

## Episode Graceful Shutdown

When duration is reached, training waits for the current episode to finish before stopping, then saves the model cleanly.

## Scenarios

| Scenario | Config File | doom_skill |
|----------|-------------|-------------|
| `deadly_corridor` | `deadly_corridor.cfg` | 5 |
| `e1m1` | `e1m1.cfg` | 3 |

## Environment Variables

```bash
TELEGRAM_BOT_TOKEN=<your_bot_token>
TELEGRAM_CHAT_ID=<your_chat_id>
```

Defaults are hardcoded in `src/notify.py` for local convenience — override via env vars to use your own bot.

Defaults are hardcoded in `src/notify.py` — override via env vars if needed.

## Project Structure

```
ludicrous-speed/
├── src/
│   ├── train.py       # Training loop + callbacks + Telegram
│   ├── play.py        # Record trained agent playing
│   ├── config.py      # Scenario + PPO hyperparameters
│   ├── env.py         # VizDoom environment wrapper
│   └── notify.py      # Telegram notifier
├── scripts/
│   └── start_training.sh [duration]  # Training launcher
├── runs/              # Output: models, stats, tensorboard
│   └── <scenario>/stats.json  # Cumulative stats per scenario
├── _vizdoom/          # VizDoom .cfg + Doom IWADs
└── requirements.txt
```

## Key Concepts

- **Cumulative Stats**: `runs/<scenario>/stats.json` persists across all runs (total_training_min, total_episodes, best_reward)
- **Graceful Shutdown**: SIGTERM finishes current episode then exits
- **Temp Video Cleanup**: Videos in `/tmp/` deleted 1 hour after sending to Telegram
- **PPO + CNN**: Raw RGB frames → CNN → PPO policy

## Verify Installation

```bash
python -c "from vizdoom import VizDoomEnv; print('VizDoom OK')"
python -c "from stable_baselines3 import PPO; print('Stable-Baselines3 OK')"
python -c "import cv2; print('OpenCV OK')"
```
