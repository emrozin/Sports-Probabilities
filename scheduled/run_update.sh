#!/bin/zsh
# Runs the whole update (python update.py --record) and adds its output to logs/update.log.
# Extra arguments are passed on (--late: only the late-news steps).
# Started by the scheduled jobs (install.sh); can also be run by hand: zsh scheduled/run_update.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') update started $* ==="
  ./.venv/bin/python update.py --record "$@"
  code=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') finished (exit code $code: 0 means every step worked) ==="
  echo
} >> logs/update.log 2>&1
# keep only the most recent part of the log
tail -n 4000 logs/update.log > logs/update.tmp && mv logs/update.tmp logs/update.log
