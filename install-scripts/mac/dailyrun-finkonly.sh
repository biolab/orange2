#!/bin/bash
#
# Should be run as: sudo ./dailyrun-finkonly.sh
#

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if ! mount | grep -q /Volumes/fink; then
	mkdir -p /Volumes/fink/
	/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) fink@biolab.si: /Volumes/fink/
fi
if ! mount | grep -q /Volumes/download; then
	mkdir -p /Volumes/download/
	/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) download@biolab.si: /Volumes/download/
fi

/Users/ailabc/fink-daily-build.sh &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?
echo "Orange (fink) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
cat /private/tmp/fink-daily-build.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
