# DoesItRunDoom? 🏎️💨

**Will the AI actually run Doom?**

A reinforcement learning project that trains an AI to play Doom using VizDoom + Stable Baselines3 (PPO). The agent learns from pixel input — no hand-coded behavior, just rewards.

**Current Scenario:** Deadly Corridor — navigate a corridor, survive 6 enemies, reach the vest.

## Quick Start

```bash
cd /home/dobby/ludicrous-speed

# Install dependencies
pip install -r requirements.txt

# Train 1 episode
./ludicrous.sh start

# Continue next episode
./ludicrous.sh continue
```

## Commands

| Command | Description |
|---------|-------------|
| `/doom_start` | Train 1 episode → Telegram notification + video |
| `/doom_continue` | Continue with next episode |
| `/doom_stop` | Stop training |
| `/doom_status` | Show current status |
| `/doom_video` | Record 2min gameplay video |

## How it Works

1. **DoomEnv** wraps VizDoom as a Gymnasium environment
2. **PPO (CnnPolicy)** learns from RGB screen input
3. **Episode-based training** — one episode at a time
4. **Notifications** sent to Telegram when episodes complete
5. **Videos** recorded headlessly with OpenCV

## Tech Stack

- VizDoom 1.3.0 — Doom engine for RL
- Stable Baselines3 2.8.0 — PPO implementation
- OpenCV 4.x — Headless video capture
- PyTorch — CNN backend
- Telegram Bot API — Notifications

## Project Structure

```
src/
├── doom_env.py       # Gymnasium env (VizDoom wrapper)
├── train_one.py      # Train for 1 episode
├── record_video.py   # Headless video recording
├── notify_done.py    # Telegram notifications
└── status.py        # Status checker
```

## Deadly Corridor Scenario

| Config | Value |
|--------|-------|
| Buttons | 7 (move, turn, shoot) |
| Objective | Reach vest, survive enemies |
| Episode Timeout | 2100 tics |
| Doom Skill | 5 (hardest) |
| Reward | +dX toward vest, -dX away, -100 death |

## Resources

- **GitHub:** https://github.com/KrabbiAI/DoesItRunDoom
- **VizDoom Docs:** https://vizdoom.farama.org
- **SB3 Docs:** https://stable-baselines3.readthedocs.io
