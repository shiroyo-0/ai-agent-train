#!/bin/bash
# Combined runner: starts continuous training + auto-push in parallel
# Usage: ./run_training.sh [--interval 30] [--push-every 3] [--model ollama/llama3]

TRAIN_INTERVAL=30      # minutes between training cycles
PUSH_INTERVAL=3        # hours between git pushes
MODEL=""
BRANCH="training-results"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval) TRAIN_INTERVAL="$2"; shift 2;;
        --push-every) PUSH_INTERVAL="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --branch) BRANCH="$2"; shift 2;;
        *) shift;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     🚀 AI AGENT - CONTINUOUS TRAINING + AUTO-PUSH       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Training interval : Every ${TRAIN_INTERVAL} minutes               ║"
echo "║  Push interval     : Every ${PUSH_INTERVAL} hours                  ║"
echo "║  Model             : ${MODEL:-auto}                     ║"
echo "║  Branch            : ${BRANCH}                          ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Stop: Ctrl+C or kill the process                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Start auto-push in background
echo "[*] Starting auto-push daemon (every ${PUSH_INTERVAL}h)..."
"$SCRIPT_DIR/auto_push.sh" "$PUSH_INTERVAL" "$BRANCH" &
PUSH_PID=$!
echo "    PID: $PUSH_PID"

# Cleanup on exit
cleanup() {
    echo ""
    echo "[*] Shutting down..."
    kill $PUSH_PID 2>/dev/null
    # Final push before exit
    cd "$REPO_DIR"
    git add -A && git commit -m "🎓 Final training state [$(date '+%Y%m%d_%H%M')]" && git push origin "$BRANCH"
    echo "[*] Final state pushed. Goodbye!"
}
trap cleanup EXIT INT TERM

# Start continuous training in foreground
echo "[*] Starting continuous training (every ${TRAIN_INTERVAL}min)..."
echo ""

MODEL_ARG=""
if [ -n "$MODEL" ]; then
    MODEL_ARG="--model $MODEL"
fi

cd "$REPO_DIR"
python3 scripts/continuous_train.py --interval "$TRAIN_INTERVAL" $MODEL_ARG
