#!/bin/bash
# DoesItRunDoom? - CLI
# Usage: ./ludicrous.sh {start|stop|status}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PIDFILE="$PROJECT_DIR/.doom_train.pid"
LOGDIR="$PROJECT_DIR/logs"
DURATION="${2:-3600}"  # default 1 hour

get_ip() {
    hostname -I | awk '{print $1}'
}

case "$1" in
    start)
        echo "🏎️  DoesItRunDoom? — Deadly Corridor"
        echo "⏱️  Training: ${DURATION}s (~$((DURATION/60)) min)"

        # Stop existing
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            kill "$PID" 2>/dev/null
            rm -f "$PIDFILE"
        fi

        # Start tensorboard
        mkdir -p "$LOGDIR"
        tensorboard --logdir "$LOGDIR" --port 6006 --bind_all > "$LOGDIR/tensorboard.log" 2>&1 &
        TB_PID=$!
        echo "📊 TensorBoard: http://$(get_ip):6006"

        # Start training
        cd "$SCRIPT_DIR"
        python3 src/train.py --duration "$DURATION" > "$LOGDIR/training.log" 2>&1 &
        TRAIN_PID=$!
        echo "$TRAIN_PID" > "$PIDFILE"

        echo "✅ Training gestartet!"
        echo "📨 Telegram-Nachricht kommt wenn fertig (~${DURATION}s)"
        ;;

    stop)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            kill "$PID" 2>/dev/null && echo "✅ Training gestoppt" || echo "⚠️ PID nicht gefunden"
            rm -f "$PIDFILE"
        else
            echo "⚠️  Kein Training aktiv"
        fi
        pkill -f "tensorboard.*6006" 2>/dev/null && echo "✅ TensorBoard gestoppt" || true
        ;;

    status)
        echo "🏎️  DoesItRunDoom? — Status"
        echo "─" * 30

        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "✅ TRAINING LAEUFT (PID $PID)"
            else
                echo "⏸️  NICHT LAUFEND (stale PID file)"
                rm -f "$PIDFILE"
            fi
        else
            echo "⏸️  NICHT LAUFEND"
        fi

        if [ -f "$LOGDIR/training.log" ]; then
            echo ""
            echo "Letzte Zeilen:"
            tail -3 "$LOGDIR/training.log" 2>/dev/null
        fi

        echo ""
        echo "📊 TensorBoard: http://$(get_ip):6006"
        ;;

    *)
        echo "Usage: ./ludicrous.sh {start|stop|status}"
        echo ""
        echo "  start [sekunden]  — Training starten (default: 1h)"
        echo "  stop              — Training stoppen"
        echo "  status            — Status anzeigen"
        exit 1
        ;;
esac
