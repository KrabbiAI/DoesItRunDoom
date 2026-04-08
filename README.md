# DoesItRunDoom?

**Trains a PPO agent to play Doom using VizDoom + Stable-Baselines3.**

**GitHub:** https://github.com/KrabbiAI/DoesItRunDoom
**Live Stats:** Keine öffentliche URL (lokal)

## Was Es Macht

Reinforcement Learning Training Pipeline das einen AI Agenten trainiert Doom's Deadly Corridor zu spielen. Schickt Telegram Notifications, zeichnet Videos auf, und trackt kumulative Trainingszeit über alle Runs.

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8+ | Runtime |
| vizdoom | latest | Doom environment |
| stable-baselines3 | latest | PPO implementation |
| gymnasium | latest | Gym interface |
| numpy | latest | Array operations |
| opencv-python | latest | Video processing |
| requests | latest | Telegram API |

## Restore from Scratch

### 1. System Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install cmake libboost-all-dev libboost-python-dev libboost-system-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev libexpat1-dev
sudo apt install zlib1g-dev libsdl2-dev libssl-dev
```

### 2. Python Dependencies

```bash
pip install vizdoom gymnasium stable-baselines3 numpy requests opencv-python
```

### 3. Doom IWAD Files

```bash
cd /home/dobby/ludicrous-speed
mkdir -p _vizdoom

# Doom 2 (für deadly_corridor scenario):
wget https://archive.org/download/DOOM2IWADFILE/DOOM2.WAD -O _vizdoom/doom2.wad

# Doom 1 Shareware (limitierte levels, kostenlos):
wget https://archive.org/download/doom-wads/Doom%20%28v1.9%29.zip -O _vizdoom/doom1.zip
unzip _vizdoom/doom1.zip -d _vizdoom/ && mv _vizdoom/doom/doom.wad _vizdoom/doom1.wad
```

### 4. Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="<your_bot_token>"
export TELEGRAM_CHAT_ID="<your_chat_id>"
```

**Credentials Storage:** Env Vars in Shell Profile oder `.env` file.

### 5. Verify Installation

```bash
python -c "from vizdoom import VizDoomEnv; print('VizDoom OK')"
python -c "from stable_baselines3 import PPO; print('Stable-Baselines3 OK')"
python -c "import cv2; print('OpenCV OK')"
```

## Training Starten

```bash
cd /home/dobby/ludicrous-speed

# Training mit Telegram Notifications
TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." \
python -m src.train \
  --outdir runs/run_$(date +%Y_%m_%d_%H%M) \
  --duration 60 \
  --scenario deadly_corridor

# Verfügbare Szenarien: deadly_corridor, e1m1
```

## Scenarios

| Scenario | Config | doom_skill | WAD Required |
|----------|--------|------------|--------------|
| `deadly_corridor` | `deadly_corridor.cfg` | 5 | doom2.wad |
| `e1m1` | `e1m1.cfg` | 3 | doom1.wad |

## Kumulative Stats

Stats werden in `runs/<scenario>/stats.json` gespeichert und bleiben über alle Runs erhalten:

```json
{
  "total_training_min": 0,
  "total_episodes": 0,
  "total_timesteps": 0,
  "best_reward": 0
}
```

**WICHTIG:** Diese Datei NIE resetten ohne Absprache mit Sascha.

## Telegram Notifications

Training sendet Notifications bei:
- **Start**: Run Info + vorherige kumulative Stats
- **Alle 5 min**: Progress Update mit kumulativer Trainingszeit
- **Ende**: Final Stats + Video Recording startet
- **Video**: Recording Resultat (prüft ob Datei existiert)

## API Endpoints (Telegram Bot)

**Telegram Bot API Base:** `https://api.telegram.org/bot<TOKEN>/`

**Send Message:**
```
POST /sendMessage
Body: {"chat_id": "<CHAT_ID>", "text": "<message>", "parse_mode": "HTML"}
```

**Env Vars:**
- `TELEGRAM_BOT_TOKEN` — Bot Token von @BotFather
- `TELEGRAM_CHAT_ID` — Chat ID des Empfängers

## Projekt Struktur

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
│   └── <scenario>/
│       └── stats.json  # Kumulative stats
├── _vizdoom/          # VizDoom .cfg + Doom IWADs
└── requirements.txt
```

## Key Concepts

- **Kumulative Stats**: `runs/<scenario>/stats.json` bleibt über alle Runs (total_training_min, total_episodes, best_reward)
- **Graceful Shutdown**: SIGTERM beendet current episode dann exit
- **Temp Video Cleanup**: Videos in `/tmp/` werden 1 Stunde nach Telegram-Senden gelöscht
- **PPO + CNN**: Raw RGB frames → CNN → PPO policy

## Troubleshooting

**VizDoom crash beim Start:**
- Doom IWAD nicht gefunden → prüfe `_vizdoom/` Verzeichnis
- Permissions → `chmod 644 _vizdoom/*.wad`

**Telegram Notifications kommen nicht:**
- Token/Chat ID prüfen
- Bot muss Chat mit User haben (start Nachricht an Bot senden)

**Training sehr langsam:**
- GPU nutzen (PYPPIN oder CUDA) oder CPU-only erwarten
- `deadly_corridor` scenario ist compute-intensiv

## Cron Setup (Optional)

```bash
# Auf Keenetic Router:
0 3 * * * /home/dobby/ludicrous-speed/scripts/start_training.sh 480
```

## Verify Deployment

```bash
# Prüfe ob Training läuft
ps aux | grep train.py

# Letzte Telegram Message
curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | jq '.result[-1].message.text'
```
