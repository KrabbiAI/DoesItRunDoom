#!/bin/bash
# DoesItRunDoom? — Start Training (60 min)
# Usage: ./scripts/start_training.sh

SESSION_NAME="doom_train"
RUN_DIR="runs/run_$(date +%Y_%m_%d_%H%M)"

mkdir -p "$RUN_DIR"

echo "🚀 Starting Doom RL Training"
echo "📁 Run directory: $RUN_DIR"
echo "⏱️  Duration: 60 minutes"
echo "🎮 Scenario: $DOOM_SCENARIO (default: deadly_corridor)"

# Start tmux session with training
tmux new-session -d -s "$SESSION_NAME" "
cd /home/dobby/ludicrous-speed && \
python -m src.train --outdir '$RUN_DIR' --duration 60 --scenario '${DOOM_SCENARIO:-deadly_corridor}'
"

echo "✅ Training started in tmux session '$SESSION_NAME'"
echo "📊 Monitor: tensorboard --logdir runs/"
echo "🔍 Watch: tmux attach -t $SESSION_NAME"
echo "⏹️  Stop: tmux kill-session -t $SESSION_NAME"
