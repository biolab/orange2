#!/bin/bash
#
# Should be run as: sudo ./dailyrun-finkonly.sh
#

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

[ -e /Volumes/fink/ ] || { open "smb://orange@estelle.fri.uni-lj.si/fink/"; sleep 30; }

echo "Running fink-daily-build.sh" > /private/tmp/fink-daily-build.log
/Users/mitar/fink-daily-build.sh &> /private/tmp/fink-daily-build.log || cat /private/tmp/fink-daily-build.log
