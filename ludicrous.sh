#!/bin/bash
# DoesItRunDoom? - CLI
# Usage: ./ludicrous.sh {start|stop|status|continue}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="/home/dobby/ludicrous-speed/logs"
PIDFILE="/home/dobby/ludicrous-speed/ludicrous_speed.pid"
TENSORBOARD_PORT=6006

case "$1" in
    start|continue)
        echo "🏎️  DoesItRunDoom? — One episode..."

        # Stop any running training first
        if [ -f "$PIDFILE" ]; then
            PIDS=$(cat "$PIDFILE")
            for PID in $PIDS; do
                kill "$PID" 2>/dev/null
            done
            rm -f "$PIDFILE"
        fi

        SERVER_IP=$(hostname -I | awk '{print $1}')
        echo "📊 TensorBoard: http://$SERVER_IP:$TENSORBOARD_PORT"

        cd "$SCRIPT_DIR"
        python3 src/run_training.py > "$LOGDIR/training.log" 2>&1 &
        MAIN_PID=$!
        echo "$MAIN_PID" > "$PIDFILE"

        echo "✅ Started! TensorBoard: http://$SERVER_IP:$TENSORBOARD_PORT"
        echo "📨 Du bekommst Telegram-Benachrichtigung wenn Episode fertig ist."
        ;;

    stop)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            kill "$PID" 2>/dev/null && echo "✅ Gestoppt" || echo "⚠️  Prozess nicht gefunden"
            rm -f "$PIDFILE"
        else
            echo "⚠️  Kein Training aktiv"
        fi
        pkill -f "tensorboard.*6006" 2>/dev/null && echo "✅ TensorBoard gestoppt" || true
        ;;

    status)
        python3 "$SCRIPT_DIR/src/status.py"
        ;;

    *)
        echo "Usage: ./ludicrous.sh {start|stop|status|continue}"
        exit 1
        ;;
esac
