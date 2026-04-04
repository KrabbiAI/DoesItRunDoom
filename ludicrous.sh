#!/bin/bash
# Ludicrous Speed - CLI for /doom_start, /doom_stop, /doom_status
# Location: /home/dobby/ludicrous-speed/ludicrous.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="/home/dobby/ludicrous-speed/logs"
PIDFILE="/home/dobby/ludicrous-speed/ludicrous_speed.pid"
TENSORBOARD_PORT=6006

case "$1" in
    start)
        # Check if already running
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "⚠️  Already running with PID $PID"
                exit 1
            else
                rm -f "$PIDFILE"
            fi
        fi

        DURATION="${2:-600}"  # default 10 minutes (600s)
        echo "🏎️  Starting Ludicrous Speed... (${DURATION}s session)"

        # Get server IP for TensorBoard URL
        SERVER_IP=$(hostname -I | awk '{print $1}')

        # Start TensorBoard in background
        nohup tensorboard --logdir "$LOGDIR" --port $TENSORBOARD_PORT --bind_all > "$LOGDIR/tensorboard.log" 2>&1 &
        TB_PID=$!
        echo "📊 TensorBoard started (PID $TB_PID)"

        # Start training in background with timeout
        cd "$SCRIPT_DIR"
        nohup python3 src/run_training.py \
            --logdir "$LOGDIR" \
            --visible \
            --timesteps 10000000 \
            --duration "$DURATION" \
            > "$LOGDIR/training.log" 2>&1 &
        TRAIN_PID=$!
        echo "🚀 Training started (PID $TRAIN_PID)"

        echo "$TRAIN_PID $TB_PID" > "$PIDFILE"
        echo ""
        echo "✅ Ludicrous Speed is GO!"
        echo ""
        echo "📊 TensorBoard: http://$SERVER_IP:$TENSORBOARD_PORT"
        echo "⏱️  Training for ${DURATION}s (~$((DURATION/60)) min)"
        echo "🖥️  VizDoom window should appear on server display..."
        ;;

    stop)
        if [ -f "$PIDFILE" ]; then
            PIDS=$(cat "$PIDFILE")
            TRAIN_PID=$(echo "$PIDS" | awk '{print $1}')
            TB_PID=$(echo "$PIDS" | awk '{print $2}')
            if kill -0 "$TRAIN_PID" 2>/dev/null; then
                echo "🛑 Stopping training (PID $TRAIN_PID)..."
                kill "$TRAIN_PID" 2>/dev/null
                sleep 1
                kill -9 "$TRAIN_PID" 2>/dev/null
                echo "✅ Training stopped"
            fi
            if [ -n "$TB_PID" ] && kill -0 "$TB_PID" 2>/dev/null; then
                echo "🛑 Stopping TensorBoard (PID $TB_PID)..."
                kill "$TB_PID" 2>/dev/null
                echo "✅ TensorBoard stopped"
            fi
            rm -f "$PIDFILE"
        else
            echo "⚠️  Not running (no PID file)"
        fi

        # Safety cleanup
        pkill -f "tensorboard.*$TENSORBOARD_PORT" 2>/dev/null && echo "✅ TensorBoard cleanup done" || true
        pkill -f "python3.*train.py" 2>/dev/null && echo "✅ Training process cleanup done" || true
        ;;

    status)
        python3 "$SCRIPT_DIR/src/status.py"
        ;;

    video)
        echo "🎬 Recording 2min gameplay..."
        cd "$SCRIPT_DIR"
        python3 src/record_video.py --duration 120 2>&1
        ;;

    *)
        echo "Usage: ludicrous.sh {start|stop|status|video}"
        exit 1
        ;;

    *)
        echo "Usage: ludicrous.sh {start|stop|status}"
        exit 1
        ;;
esac
