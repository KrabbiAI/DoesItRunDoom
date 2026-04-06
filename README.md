# DoesItRunDoom?

**Trains a PPO agent to play Doom using VizDoom + Stable-Baselines3.**

## What This Does

Reinforcement Learning training pipeline that teaches an agent to navigate Doom's Deadly Corridor. Sends Telegram status notifications every 5 minutes, records a video at the end, and tracks cumulative training time across all runs.

## Quick Start

```bash
cd /home/dobby/ludicrous-speed

# Install Python dependencies
pip install -r requirements.txt

# Install VizDoom (needs cmake, boost, etc.)
# On Ubuntu/Debian:
sudo apt install cmake libboost-all-dev libboost-python-dev libboost-system-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev libexpat1-dev
sudo apt install zlib1g-dev SDL2-dev libssl-dev

# Get a Doom IWAD (doom2.wad) and place it in _vizdoom/
# The scenario .cfg files reference: deadly_corridor.cfg

# Start training (default: 60 min, deadly_corridor)
python -m src.train --outdir runs/my_run --duration 60 --scenario deadly_corridor
```

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
