# DoesItRunDoom? 🏎️💨

**Will the AI actually run Doom?**

Reinforcement Learning Projekt — trainiert einen AI Agent mit PPO + VizDoom, um Doom zu spielen. Der Agent lernt von Pixel-Input. Kein handgeschriebenes Verhalten. Nur Rewards.

**Szenario:** Deadly Corridor — durchquere einen Korridor, überlebe 6 Gegner, finde die Weste.

---

## Setup

```bash
cd /home/dobby/ludicrous-speed

# Dependencies installieren
pip install -r requirements.txt

# Oder direkt:
pip install vizdoom>=1.3.0 gymnasium>=0.29.0 stable-baselines3>=2.0.0 numpy>=1.24.0 requests>=2.31.0 opencv-python>=4.0.0
```

**Wichtig:** VizDoom braucht ein funktionierendes `_vizdoom/` Verzeichnis mit WAD-Files. Das `vizdoom`-Package bringt `freedoom1.wad` mit — das reicht für Deadly Corridor.

---

## Projekt starten

### Training (60 Minuten)

```bash
cd /home/dobby/ludicrous-speed
./ludicrous.sh start 3600
```

- Trainiert 60 Minuten (3600 Sekunden)
- Alle 5 Minuten ein Status-Update per Telegram
- Am Ende: Modell speichern + Video aufnehmen + zu Telegram schicken

### Training stoppen

```bash
./ludicrous.sh stop
```

### Status anzeigen

```bash
./ludicrous.sh status
```

### Video aufnehmen (vom aktuellen Modell)

```bash
cd /home/dobby/ludicrous-speed
python3 src/play.py
```

Video wird nach `/tmp/doom_playthrough.mp4` geschrieben und zu Telegram geschickt.

---

## Wie es funktioniert

```
ludicrous.sh (start/stop/status)
    └── train.py
            ├── config.py       — Scenario + PPO Hyperparameters
            ├── env.py          — ScreenOnlyWrapper (VizDoom → Gymnasium)
            ├── notify.py       — Telegram Notifications
            ├── train.py        — PPO Training Loop + Callbacks
            └── play.py        — Video Recording + Telegram Upload
```

1. **VizDoom** läuft als Gymnasium-Environment
2. **PPO (CnnPolicy)** lernt von RGB-Screen-Input
3. **TensorBoard** Logging in `runs/<run_name>/tensorboard/`
4. **Telegram** Notifications bei Status-Updates und nach dem Training
5. **Video** wird mit OpenCV aus den raw Screen-Buffers aufgezeichnet

---

## Telegram Commands (via OpenClaw Bot)

| Command | Beschreibung |
|---------|-------------|
| `/doom_start` | 60 min Training starten |
| `/doom_stop` | Training stoppen |
| `/doom_status` | Aktuellen Status zeigen |
| `/doom_video` | Video vom aktuellen Modell aufnehmen |

---

## Szenarien

### Deadly Corridor (aktuell)

| Config | Wert |
|--------|------|
| Buttons | 7 (vorwärts, rückwärts, drehen, schießen) |
| Ziel | Weste finden, Gegner überleben |
| Episode Timeout | 2100 Ticks |
| Doom Skill | 3 |
| Reward | +dX Richtung Weste, -dX weg, -100 bei Tod |

### Health Gathering (verfügbar, nicht aktiv)

```python
# In config.py: SCENARIOS["health_gathering"]
# Gymnasium ID: "VizdoomHealthGatheringSupreme-v0"
```

---

## Wichtige Dateien

```
src/
├── config.py          — SCENARIOS dict + PPO Hyperparameters
├── env.py             — ScreenOnlyWrapper (resized screen extraction)
├── notify.py          — TelegramNotifier class
├── train.py           — Training Loop + StatusCallback (5-min updates)
└── play.py            — Video Recording + Telegram Upload

runs/                  — Trainings-Runs (modelle, tensorboard logs)
_vizdoom/              — VizDoom WADs + Configs (auto-generated)
_vizdoom.ini           — VizDoom Settings
ludicrous.sh          — CLI: start / stop / status
requirements.txt       — Python Dependencies
```

---

## Troubleshooting

**NNPACK Warnings:** Normal auf CPUs ohne AVX2. Training läuft trotzdem.

**"No module named vizdoom":** `pip install vizdoom>=1.3.0`

**Training startet nicht:** Prüfe ob PID-File existiert (`cat .doom_train.pid`) und Prozess noch läuft (`kill -0 <PID>`)

**Trennlinie:** Notifications haben eine `──────────────` Trennlinie zwischen Header und Stats

**Neues Modell starten (Reset):**
```bash
rm -f runs/default/final_model.zip runs/default/monitor.csv runs/default/training_stats.json
./ludicrous.sh start 3600
```

**Video schwarz:** Raw VizDoom Frames sind sehr dunkel → `play.py` macht 2.5x brightness boost automatisch.

**Telegram Notifications kommen nicht:** `src/notify.py` Credentials prüfen — aktuell hardcoded im Code.

---

## Tech Stack

- **VizDoom 1.3.0** — Doom Engine für RL
- **Stable Baselines3 2.x** — PPO Implementation
- **Gymnasium** — Environment Wrapper Standard
- **OpenCV 4.x** — Headless Video Capture
- **PyTorch** — CNN Backend
- **Telegram Bot API** — Notifications

---

## Ressourcen

- **GitHub:** https://github.com/KrabbiAI/DoesItRunDoom
- **VizDoom Docs:** https://vizdoom.farama.org
- **SB3 Docs:** https://stable-baselines3.readthedocs.io
- **Deadly Corridor Paper:** https://arxiv.org/abs/1605.09128
