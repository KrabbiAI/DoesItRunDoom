# Ludicrous Speed 🏎️💨
**To go where no AI has gone before.**

Doom RL — VizDoom + Stable Baselines3 PPO, AI lernt Doom spielen.

## Setup

```bash
cd /home/dobby/ludicrous-speed
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- VizDoom (C++ build needed — `apt install vizdoom` on Linux, or build from source)
- CUDA GPU (optional, runs on CPU too)

### VizDoom Install (Linux)

```bash
# Option 1: From source (recommended for headless/server)
git clone https://github.com/mwydmuch/ViZDoom
cd ViZDoom && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
pip install vizdoom

# Option 2: Check if already installed
python -c "import vizdoom; print(vizdoom.vizdoom_path())"
```

## Usage

### Commands via Telegram

```
/doom_start  — Start training + TensorBoard
/doom_stop   — Stop training
/doom_status — Show status
```

### Manual

```bash
cd /home/dobby/ludicrous-speed

# Start training + TensorBoard
./ludicrous.sh start

# Check status
./ludicrous.sh status

# Stop
./ludicrous.sh stop
```

## TensorBoard

While training:
```bash
tensorboard --logdir ./logs --port 6006
# → http://localhost:6006
```

Metrics tracked:
- `rollout/ep_rew_mean` — average episode reward
- `rollout/ep_len_mean` — average episode length
- `train/loss` — policy loss
- `train/entropy` — action entropy

## Project Structure

```
ludicrous-speed/
├── src/
│   ├── doom_env.py      # Gymnasium wrapper for VizDoom
│   ├── train.py         # PPO training with callbacks
│   ├── play.py          # Inference + video recording
│   ├── run_training.py  # Runner: TensorBoard + training
│   └── status.py        # Status checker
├── models/              # Trained models (.zip)
├── logs/                # TensorBoard logs
├── videos/              # Recorded gameplay
├── ludicrous.sh         # CLI wrapper (start/stop/status)
└── requirements.txt
```

## Scenarios

Available VizDoom scenarios:
- `basic` — Find and kill the enemy (default)
- `deadly_corridor` — Navigate a corridor, collect items
- `defend_the_center` — Survive waves
- `deathmatch` — Multi-player style
- `take_cover` — Use obstacles

```bash
./ludicrous.sh start --scenario deadly_corridor
```

## Architecture

- **Algorithm:** PPO (Proximal Policy Optimization)
- **Policy:** CNN (CnnPolicy) — raw pixel input
- **Network:** [256, 256] hidden layers
- **Observations:** 160x120 RGB screen
- **Actions:** 8 discrete (turn L/R, move F/B, strafe L/R, shoot, use)
- **Reward:** +1/step alive, +10/kill, -100/death, +100/victory

## Play a Trained Model

```bash
python src/play.py --model models/doom_ppo_ludicrous.zip --episodes 3 --record
# Recordings saved to videos/
```
