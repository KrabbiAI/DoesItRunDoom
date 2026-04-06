#!/bin/bash
# DoesItRunDoom? - CLI
# Usage: ./ludicrous.sh {start|stop|status} [scenario] [sekunden]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
PIDFILE="$PROJECT_DIR/.doom_train.pid"
LOGDIR="$PROJECT_DIR/logs"

get_ip() {
    hostname -I | awk '{print $1}'
}

# Parse arguments: action, scenario, duration
ACTION="${1:-}"
SCENARIO="${2:-deadly_corridor}"
DURATION="${3:-3600}"

case "$ACTION" in
    start)
        # Check if scenario is valid
        if [ ! -d "$PROJECT_DIR/runs/$SCENARIO" ]; then
            mkdir -p "$PROJECT_DIR/runs/$SCENARIO"
        fi
        OUTDIR="$PROJECT_DIR/runs/$SCENARIO"
        MODEL_PATH="$OUTDIR/final_model"
        
        echo "🏎️  DoesItRunDoom? — $SCENARIO"
        echo "⏱️  Training: ${DURATION}s (~$(($DURATION / 60)) min)"
        echo "📊 TensorBoard: http://$(get_ip):6006"
        echo ""
        if [ -f "${MODEL_PATH}.zip" ]; then
            echo "⚠️  Modell existiert bereits — wird weitertrainiert:"
            echo "   📁 Folder: $OUTDIR"
            echo "   🧠 Modell: ${MODEL_PATH}.zip"
        else
            echo "🆕 Neues Modell wird erstellt:"
            echo "   📁 Folder: $OUTDIR"
        fi
        echo ""
        echo "✅ Bestätige mit 'yes' um zu starten:"
        read CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            echo "❌ Abgebrochen."
            exit 0
        fi

        # Stop existing graceful first
        if [ -f "$PIDFILE" ]; then
            OLD_PID=$(cat "$PIDFILE")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "🛑 Stoppe laufendes Training (PID $OLD_PID)..."
                kill -TERM "$OLD_PID" 2>/dev/null
                sleep 5
                if kill -0 "$OLD_PID" 2>/dev/null; then
                    kill -9 "$OLD_PID" 2>/dev/null
                fi
            fi
            rm -f "$PIDFILE"
        fi

        # Start tensorboard
        mkdir -p "$LOGDIR"
        pkill -f "tensorboard.*6006" 2>/dev/null || true
        tensorboard --logdir "$PROJECT_DIR/runs" --port 6006 --bind_all > "$LOGDIR/tensorboard.log" 2>&1 &
        TB_PID=$!
        echo "📊 TensorBoard: http://$(get_ip):6006"

        # Start training
        cd "$SCRIPT_DIR"
        python3 src/train.py --scenario "$SCENARIO" --duration $((DURATION / 60)) --outdir "$OUTDIR" > "$LOGDIR/training.log" 2>&1 &
        TRAIN_PID=$!
        echo "$TRAIN_PID" > "$PIDFILE"

        echo "✅ Training gestartet!"
        echo "📨 Telegram-Nachricht kommt wenn fertig (~$(($DURATION / 60)) min)"
        ;;

    stop)
        echo "🛑 Stoppe Training..."
        # 1. Graceful SIGTERM first (up to 15s)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "⏳ Graceful SIGTERM an PID $PID..."
                kill -TERM "$PID" 2>/dev/null
                for i in $(seq 1 15); do
                    sleep 1
                    if ! kill -0 "$PID" 2>/dev/null; then
                        echo "✅ Training sauber gestoppt"
                        break
                    fi
                done
                if kill -0 "$PID" 2>/dev/null; then
                    echo "⚠️  Graceful timeout — SIGKILL"
                    kill -9 "$PID" 2>/dev/null
                fi
            else
                echo "⚠️  PID nicht aktiv"
            fi
            rm -f "$PIDFILE"
        else
            echo "⚠️  Kein Training aktiv"
        fi
        pkill -9 -f "vizdoom" 2>/dev/null && echo "🧹 Zombie VizDooms gekillt" || true
        pkill -f "tensorboard.*6006" 2>/dev/null && echo "✅ TensorBoard gestoppt" || true
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start "$2" "$3"
        ;;

    status)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "✅ Training aktiv (PID $PID)"
            else
                echo "⚠️  PID existiert aber Prozess tot"
            fi
        else
            echo "⚠️  Kein Training aktiv"
        fi
        echo ""
        echo "📊 TensorBoard: http://$(get_ip):6006"
        ;;

    *)
        echo "Usage: ./ludicrous.sh {start|stop|restart|status} [scenario] [sekunden]"
        echo "  start   [scenario] [sekunden]  — Training starten"
        echo "  stop                           — Training stoppen"
        echo "  restart [scenario] [sekunden]  — Stoppen + neu starten"
        echo "  status                         — Status anzeigen"
        echo ""
        echo "Scenarios: deadly_corridor, e1m1"
        ;;
esac
