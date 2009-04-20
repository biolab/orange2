#!/bin/bash
#
# Should be run as: sudo ./dailyrun-bundleonly.sh
#

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

[ -e /Volumes/download/ ] || { open "smb://orange@estelle.fri.uni-lj.si/download/"; sleep 30; }

echo "Running bundle-daily-build.sh" > /private/tmp/bundle-daily-build.log
/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log || cat /private/tmp/bundle-daily-build.log
