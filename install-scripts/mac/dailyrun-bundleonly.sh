#!/bin/bash
#
# Should be run as: sudo ./dailyrun-bundleonly.sh
#

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if [ ! -e /Volumes/download/ ]; then
	mkdir -p /Volumes/download/
	/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) download@biolab.si: /Volumes/download/
fi

/Users/ailabc/bundle-daily-build.sh &> /private/tmp/bundle-daily-build.log
EXIT_VALUE=$?
echo "Orange (bundle) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
cat /private/tmp/bundle-daily-build.log >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running bundle-daily-build.sh failed"

# Zero exit value
true
