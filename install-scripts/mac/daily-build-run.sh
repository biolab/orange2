#!/bin/bash

REMOTE=download@biolab.si:files
PUBLISH_DIR="$HOME"/Volumes/download
LOG="$PUBLISH_DIR"/buildLogs/osx/daily-build-osx-app.log

mkdir -p "$PUBLISH_DIR"

if { ! mount -t fusefs | grep "$PUBLISH_DIR" > /dev/null; } then
    sshfs -o reconnect,uid=$(id -u),gid=$(id -g) \
        -o workaround=nonodelay:rename \
        "$REMOTE" \
        "$PUBLISH_DIR"
    sleep 5
fi

curl --silent --fail --location --max-redirs 3 -O https://github.com/biolab/orange/raw/master/install-scripts/mac/daily-build.sh

mkdir -p $(dirname "$LOG")

./daily-build.sh --bootstrap "$PUBLISH_DIR" 2>&1 | tee "$LOG".tmp > daily-build.log
EXIT_VALUE=$?

echo "Orange OSX app build [$EXIT_VALUE]
$(date)
$(cat daily-build.log)" > "$LOG"

rm "$LOG".tmp

sync

diskutil unmount "$PUBLISH_DIR"
