# DoesItRunDoom? — Project Sketch

## What is this?

**DoesItRunDoom?** — A reinforcement learning project that trains an AI to play Doom using VizDoom + Stable Baselines3 (PPO).

The agent learns to navigate combat scenarios, collect items, and survive — purely from pixel input and rewards.

---

## Architecture

```
DoomGame (VizDoom) ← Screen Buffer (RGB 160x120)
                         ↓
                   DoomEnv (Gymnasium wrapper)
                         ↓
                   PPO (CnnPolicy) — 256x256 hidden layers
                         ↓
                   Action: 7 discrete actions
                         ↓
                   Reward Shaping
```

---

## Scenario: Deadly Corridor

**Level:** A corridor with 6 shooting monsters. Goal: reach the green vest at the end.

**Configuration (from deadly_corridor.cfg):**
- 7 Buttons: MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK
- 1 Game Variable: HEALTH
- Episode Timeout: 2100 tics
- Doom Skill: 5 (hardest)

**Rewards:**
- +dX for getting CLOSER to the vest (positive distance change)
- -dX for getting FURTHER from the vest (negative distance change)
- -100 for death

**Observation Space:**
- RGB screen: 160×120×3 (channel-first for PyTorch)
- Game variable: HEALTH (1 scalar)

**Action Space:**
- Discrete: 7 actions (one per button combination)

---

## Action Mapping (CORRECT for Deadly Corridor)

The 7 available buttons map to these discrete actions:

| Action ID | Button | Description |
|-----------|--------|-------------|
| 0 | TURN_LEFT | Rotate left |
| 1 | TURN_RIGHT | Rotate right |
| 2 | MOVE_FORWARD | Walk forward |
| 3 | MOVE_BACKWARD | Walk backward |
| 4 | MOVE_LEFT | Strafe left |
| 5 | MOVE_RIGHT | Strafe right |
| 6 | ATTACK | Shoot |

Note: Unlike other scenarios, deadly_corridor requires TURN_LEFT/RIGHT for navigation, not just strafe.

---

## Reward Shaping

VizDoom's internal reward system already provides:
- +dX when moving toward the vest
- -dX when moving away
- -100 on death

We add:
- +1 per step survived (living reward to encourage survival)
- Health check: episode ends early if HEALTH ≤ 0

---

## Training Loop

### Episode-based Training

Each run = 1 complete episode:

1. Load existing model (if available)
2. Run PPO.learn() for 1 episode
3. Save model to `doom_ppo_latest.zip`
4. Record 2 min gameplay video
5. Send Telegram notification with stats + video

Commands:
```
/doom_start    → Start 1 episode → notify → video
/doom_continue → Continue training next episode
/doom_stop     → Stop training
/doom_status    → Show status
/doom_video    → Record gameplay video
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Doom Engine | VizDoom 1.3.0 |
| RL Framework | Stable Baselines3 2.8.0 |
| Gym Interface | gymnasium + custom DoomEnv |
| Video Capture | OpenCV 4.x (headless frame capture) |
| TensorBoard | PyTorch SummaryWriter |
| Notifications | Telegram Bot API |

---

## Project Structure

```
doesitrumdoom/
├── src/
│   ├── doom_env.py        # Gymnasium env wrapper for VizDoom
│   ├── train_one.py       # Train for 1 episode
│   ├── record_video.py    # Headless video recording
│   ├── notify_done.py     # Telegram notification sender
│   └── status.py          # Status checker
├── models/                # Trained .zip models
├── logs/
│   └── tensorboard/       # TB event files
├── videos/                # Recorded gameplay
├── requirements.txt
└── README.md
```

---

## Known Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| VizDoom window crashes on server | No GPU/display | Run headless |
| Episode ends in 0s | Buttons not set correctly | Use load_config() AFTER setting WAD paths |
| Buttons empty in env | load_config() overrides paths | Set paths BEFORE load_config() |
| No live viewing | Wayland on server | Accept headless; videos show gameplay |
| TensorBoard callback missing | SB3 2.8 moved it | Use torch.utils.tensorboard.SummaryWriter directly |

---

## Configuration Checklist (per scenario)

When switching scenarios, verify:

- [ ] `doom_scenario_path` → correct .wad
- [ ] `doom_skill` → difficulty
- [ ] `episode_timeout` → max length
- [ ] `death_penalty` → negative reward on death
- [ ] `available_buttons` → matches action array size
- [ ] `available_game_variables` → HEALTH, AMMO, etc.
- [ ] `screen_resolution` → 160x120 (matches CNN input)
- [ ] Action space matches number of available buttons

---

## Media

- **GitHub:** https://github.com/KrabbiAI/DoesItRunDoom
- **Training:** Episode-based, ~2-10 min per episode depending on scenario
- **Output:** Telegram notifications with gameplay videos

---

## TODO / Next Steps

- [ ] Test MultiBinary action space (allow simultaneous button presses)
- [ ] Add more reward shaping (health pickup bonus, kill bonus)
- [ ] Try other scenarios: health_gathering_supreme, defend_the_center
- [ ] Track episode statistics over time (reward curve)
- [ ] Save best model separately from latest model
