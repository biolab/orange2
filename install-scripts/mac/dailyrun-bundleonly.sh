#!/bin/bash
#
# Should be run as: sudo ./dailyrun-bundleonly.sh
#

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

[ -e /Volumes/download/ ] || { open "smb://orange@193.2.72.35/download/"; sleep 30; }

/Users/ailabc/bundle-daily-build.sh &> /private/tmp/bundle-daily-build.log
EXIT_VALUE=$?
echo "Orange (bundle) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
cat /private/tmp/bundle-daily-build.log >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running bundle-daily-build.sh failed"

# Zero exit value
true
