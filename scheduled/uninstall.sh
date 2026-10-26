#!/bin/zsh
# Removes the automatic updates set up by install.sh.
for LABEL in com.sportsprobabilities.update com.sportsprobabilities.late; do
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null
  rm -f "$PLIST"
done
echo "Removed: the update no longer runs on its own."
