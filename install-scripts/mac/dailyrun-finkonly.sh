#!/bin/bash
#
# Should be run as: sudo ./dailyrun-finkonly.sh
#

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

./mount-dirs.sh

/Users/ailabc/fink-daily-build.sh &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?

./mount-dirs.sh

echo "Orange (fink) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
cat /private/tmp/fink-daily-build.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
