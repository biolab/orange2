#!/bin/bash
#
# Should be run as: sudo ./dailyrun-finkonly.sh
#

test -r /sw/bin/init.sh && . /sw/bin/init.sh

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`
ARCH=`perl -MFink::FinkVersion -e 'print Fink::FinkVersion::get_arch'`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

/Users/ailabc/fink-daily-build.sh &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

echo "Orange (fink $MAC_VERSION $ARCH) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
cat /private/tmp/fink-daily-build.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
