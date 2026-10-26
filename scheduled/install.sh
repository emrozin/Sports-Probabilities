#!/bin/zsh
# Sets up the automatic update on this Mac: python update.py --record every day at 10:07 am and
# 5:37 pm, and the late-news steps (python update.py --record --late: the NBA injury report,
# predictions, lines and bet log) at 6:37 pm, all the Mac's time; output in logs/update.log. If the
# Mac is asleep or off at those times, a missed run happens once when it next wakes up.
#   Install:    zsh scheduled/install.sh
#   Run now:    launchctl kickstart gui/$(id -u)/com.sportsprobabilities.update
#   Remove:     zsh scheduled/uninstall.sh
set -e
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.sportsprobabilities.update"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LATE_LABEL="com.sportsprobabilities.late"
LATE_PLIST="$HOME/Library/LaunchAgents/$LATE_LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT/logs"
if [ ! -x "$PROJECT/.venv/bin/python" ]; then
  echo "No .venv found in $PROJECT: run this from the project that has its .venv set up."
  exit 1
fi
cat > "$PLIST" <<PLIST_END
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$PROJECT/scheduled/run_update.sh</string></array>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>10</integer><key>Minute</key><integer>7</integer></dict>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>37</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$PROJECT/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/launchd.log</string>
</dict>
</plist>
PLIST_END
cat > "$LATE_PLIST" <<PLIST_END
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LATE_LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/zsh</string><string>$PROJECT/scheduled/run_update.sh</string><string>--late</string></array>
  <key>StartCalendarInterval</key>
  <dict><key>Hour</key><integer>18</integer><key>Minute</key><integer>37</integer></dict>
  <key>StandardOutPath</key><string>$PROJECT/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/launchd.log</string>
</dict>
</plist>
PLIST_END
for p in "$PLIST" "$LATE_PLIST"; do
  launchctl bootout "gui/$(id -u)" "$p" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$p"
done
echo "Installed: the update runs every day at 10:07 am and 5:37 pm, and the late NBA news at 6:37 pm."
echo "Its output goes to $PROJECT/logs/update.log"
echo "To run it right now:  launchctl kickstart gui/$(id -u)/$LABEL"
