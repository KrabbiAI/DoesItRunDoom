#!/bin/bash
# DoesItRunDoom? - CLI
# Usage: ./ludicrous.sh {start|stop|status|continue}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
LOGDIR="$PROJECT_DIR/logs"
PIDFILE="$PROJECT_DIR/ludicrous_speed.pid"
TENSORBOARD_PORT=6006

# Get server IP
get_ip() {
    hostname -I | awk '{print $1}'
}

case "$1" in
    start|continue)
        echo "🏎️  DoesItRunDoom? — Deadly Corridor..."

        # Stop any running training first
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            kill "$PID" 2>/dev/null
            rm -f "$PIDFILE"
        fi

        SERVER_IP=$(get_ip)
        echo "📊 TensorBoard: http://$SERVER_IP:$TENSORBOARD_PORT"

        cd "$SCRIPT_DIR"
        python3 src/run_training.py > "$LOGDIR/training.log" 2>&1 &
        MAIN_PID=$!
        echo "$MAIN_PID" > "$PIDFILE"

        echo "✅ Started! Episode läuft..."
        echo "📨 Telegram-Benachrichtigung kommt wenn Episode fertig ist."
        echo "📊 TensorBoard: http://$SERVER_IP:$TENSORBOARD_PORT"
        ;;

    stop)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            kill "$PID" 2>/dev/null && echo "✅ Training gestoppt" || echo "⚠️ Prozess nicht gefunden"
            rm -f "$PIDFILE"
        else
            echo "⚠️  Kein Training aktiv"
        fi
        pkill -f "tensorboard.*$TENSORBOARD_PORT" 2>/dev/null && echo "✅ TensorBoard gestoppt" || true
        ;;

    status)
        python3 "$SCRIPT_DIR/src/status.py"
        ;;

    *)
        echo "Usage: ./ludicrous.sh {start|stop|status|continue}"
        echo ""
        echo "Commands:"
        echo "  start, continue  — Train 1 episode → notify → video"
        echo "  stop              — Stop training"
        echo "  status            — Show status"
        exit 1
        ;;
esac
