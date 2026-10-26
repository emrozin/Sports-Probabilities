#!/bin/zsh
# Shows whether the automatic update is installed and how the last runs went.
LABEL="com.sportsprobabilities.update"
cd "$(dirname "$0")/.." || exit 1
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  echo "The automatic update is installed (10:07 am and 5:37 pm daily)."
else
  echo "The automatic update is not installed: zsh scheduled/install.sh"
fi
if launchctl print "gui/$(id -u)/com.sportsprobabilities.late" >/dev/null 2>&1; then
  echo "The late NBA news run is installed (6:37 pm daily)."
else
  echo "The late NBA news run is not installed: zsh scheduled/install.sh"
fi
if [ -f logs/update.log ]; then
  echo
  echo "Last runs:"
  grep "===" logs/update.log | tail -n 6
  echo
  echo "Steps that failed in the last run (if any):"
  awk '/update started/{block=""} {block=block"\n"$0} END{print block}' logs/update.log | grep -i "failed" || echo "  none"
fi
