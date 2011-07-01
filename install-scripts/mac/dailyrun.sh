#!/bin/bash
#
# Should be run as: sudo ./dailyrun.sh
#

test -r /sw/bin/init.sh && . /sw/bin/init.sh

STABLE_REVISION_1=`svn info --non-interactive http://orange.biolab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_1" ] || exit 1
STABLE_REVISION_2=`svn info --non-interactive http://orange.biolab.si/svn/orange/externals/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_2" ] || exit 1

if [[ $STABLE_REVISION_1 -gt $STABLE_REVISION_2 ]]; then
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

if [[ $DAILY_REVISION_1 -gt $DAILY_REVISION_2 ]]; then
    DAILY_REVISION=$DAILY_REVISION_1
else
    DAILY_REVISION=$DAILY_REVISION_2
fi

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log
EXIT_VALUE=$?

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

echo "Orange (bundle) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
cat /private/tmp/bundle-daily-build.log >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running bundle-daily-build.sh failed"

/Users/ailabc/bundle-64bit-daily-build.sh $DAILY_REVISION &> /private/tmp/bundle-64bit-daily-build.log
EXIT_VALUE=$?

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

echo "Orange (bundle-64bit) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-64bit-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-64bit-daily-build.log"
cat /private/tmp/bundle-64bit-daily-build.log >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-64bit-daily-build.log"
(($EXIT_VALUE)) && echo "Running bundle-64bit-daily-build.sh failed"

/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?

/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }

echo "Orange (fink) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
cat /private/tmp/fink-daily-build.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
