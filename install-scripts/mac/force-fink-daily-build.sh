#!/bin/bash
#
# Should be run as: sudo ./force-fink-daily-build.sh
#
# Supplies fink daily build with latest revision so that source archives are also built
#

STABLE_REVISION_1=`svn info --non-interactive http://orange.biolab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_1" ] || exit 1
STABLE_REVISION_2=`svn info --non-interactive http://orange.biolab.si/svn/orange/externals/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_2" ] || exit 1

if [[ $STABLE_REVISION_1 > $STABLE_REVISION_2 ]]; then
    STABLE_REVISION=$STABLE_REVISION_1
else
    STABLE_REVISION=$STABLE_REVISION_2
fi

DAILY_REVISION_1=`svn info --non-interactive http://orange.biolab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION_1" ] || exit 1
DAILY_REVISION_2=`svn info --non-interactive http://orange.biolab.si/svn/orange/externals/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION_2" ] || exit 1

if [[ $DAILY_REVISION_1 > $DAILY_REVISION_2 ]]; then
    DAILY_REVISION=$DAILY_REVISION_1
else
    DAILY_REVISION=$DAILY_REVISION_2
fi

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if ! mount | grep -q /Volumes/fink; then
	mkdir -p /Volumes/fink/
	/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) fink@biolab.si: /Volumes/fink/
fi
if ! mount | grep -q /Volumes/download; then
	mkdir -p /Volumes/download/
	/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) download@biolab.si: /Volumes/download/
fi

/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION

# &> /private/tmp/fink-daily-build.log
#EXIT_VALUE=$?
#echo "Orange (fink) [$EXIT_VALUE]" > /Volumes/download/buildLogs/osx/fink-daily-build.log
#cat /private/tmp/fink-daily-build.log >> /Volumes/download/buildLogs/osx/fink-daily-build.log
#(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
