#!/bin/zsh
# Removes the automatic update set up by install.sh.
LABEL="com.sportsprobabilities.update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null
rm -f "$PLIST"
echo "Removed: the update no longer runs on its own."
