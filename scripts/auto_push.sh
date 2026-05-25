#!/bin/bash
# Auto-push training results to GitHub every N hours
# Usage: ./auto_push.sh [interval_hours] [branch_name]

INTERVAL_HOURS=${1:-3}
BRANCH=${2:-"training-results"}
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "╔══════════════════════════════════════════════════════╗"
echo "║       📤 AUTO-PUSH TO GITHUB                        ║"
echo "║                                                      ║"
echo "║  Interval: Every ${INTERVAL_HOURS} hours                          ║"
echo "║  Branch: ${BRANCH}                          ║"
echo "║  Repo: ${REPO_DIR}  ║"
echo "╚══════════════════════════════════════════════════════╝"

cd "$REPO_DIR"

# Ensure branch exists
git checkout -B "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo ""
    echo "[$TIMESTAMP] Checking for changes to push..."

    # Add training outputs
    git add -A data/training/ data/logs/ models/ 2>/dev/null
    git add -A src/ configs/ scripts/ 2>/dev/null

    # Check if there are changes
    if git diff --cached --quiet && git diff --quiet; then
        echo "  No changes to push."
    else
        # Commit
        CYCLE_COUNT=$(find data/logs -name "training_cycle_*.json" 2>/dev/null | wc -l)
        COMMIT_MSG="🎓 Training update: ${CYCLE_COUNT} cycles completed [$(date '+%Y%m%d_%H%M')]"

        git add -A
        git commit -m "$COMMIT_MSG"

        # Push
        echo "  Pushing to origin/${BRANCH}..."
        if git push -u origin "$BRANCH" 2>&1; then
            echo "  ✓ Pushed successfully at $TIMESTAMP"
        else
            echo "  ✗ Push failed. Will retry next interval."
        fi
    fi

    echo "  ⏳ Next push in ${INTERVAL_HOURS} hours..."
    sleep $((INTERVAL_HOURS * 3600))
done
