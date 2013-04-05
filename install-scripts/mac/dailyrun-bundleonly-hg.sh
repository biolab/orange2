#!/bin/bash
#
# Should be run as: sudo ./dailyrun-bundleonly-hg.sh
#

export PATH=$HOME/bin:$PATH

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

/Users/ailabc/bundle-daily-build-hg.sh /private/tmp /Volumes/download &> /private/tmp/bundle-daily-build-hg.log
EXIT_VALUE=$?

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

LOG_FILE="/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build-hg.log"

echo "Orange (bundle $MAC_VERSION) [$EXIT_VALUE]" > $LOG_FILE
date >> $LOG_FILE
cat /private/tmp/bundle-daily-build-hg.log >> $LOG_FILE
(($EXIT_VALUE)) && echo "Running bundle-daily-build-hg.sh failed"

# Zero exit value
true
