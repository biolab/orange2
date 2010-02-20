#!/bin/bash
#
# Should be run as: sudo ./dailyrun-finkonly.sh
#

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

[ -e /Volumes/fink/ ] || { open "smb://orange@193.2.72.35/fink/"; sleep 30; }
[ -e /Volumes/download/ ] || { open "smb://orange@193.2.72.35/download/"; sleep 30; }

/Users/ailabc/fink-daily-build.sh &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?
echo "Orange (fink) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
cat /private/tmp/fink-daily-build.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
