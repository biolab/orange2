#!/bin/bash
#
# Should be run as: sudo ./dailyrun.sh
#

STABLE_REVISION_1=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_1" ] || exit 1
STABLE_REVISION_2=`svn info --non-interactive http://www.ailab.si/svn/orange/externals/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$STABLE_REVISION_2" ] || exit 1

if [[ $STABLE_REVISION_1 > $STABLE_REVISION_2 ]]; then
    STABLE_REVISION=$STABLE_REVISION_1
else
    STABLE_REVISION=$STABLE_REVISION_2
fi

DAILY_REVISION_1=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION_1" ] || exit 1
DAILY_REVISION_2=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION_2" ] || exit 1

if [[ $DAILY_REVISION_1 > $DAILY_REVISION_2 ]]; then
    DAILY_REVISION=$DAILY_REVISION_1
else
    DAILY_REVISION=$DAILY_REVISION_2
fi

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

[ -e /Volumes/fink/ ] || { open "smb://orange@193.2.72.35/fink/"; sleep 30; }
[ -e /Volumes/download/ ] || { open "smb://orange@193.2.72.35/download/"; sleep 30; }

/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log
EXIT_VALUE=$?
echo "Orange (bundle) [$EXIT_VALUE]" > /Volumes/download/buildLogs/osx/bundle-daily-build.log
cat /private/tmp/bundle-daily-build.log >> /Volumes/download/buildLogs/osx/bundle-daily-build.log
(($EXIT_VALUE)) && echo "Running bundle-daily-build.sh failed"

/Users/ailabc/bundle-64bit-daily-build.sh $DAILY_REVISION &> /private/tmp/bundle-64bit-daily-build.log
EXIT_VALUE=$?
echo "Orange (bundle-64bit) [$EXIT_VALUE]" > /Volumes/download/buildLogs/osx/bundle-64bit-daily-build.log
cat /private/tmp/bundle-64bit-daily-build.log >> /Volumes/download/buildLogs/osx/bundle-64bit-daily-build.log
(($EXIT_VALUE)) && echo "Running bundle-64bit-daily-build.sh failed"

/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/fink-daily-build.log
EXIT_VALUE=$?
echo "Orange (fink) [$EXIT_VALUE]" > /Volumes/download/buildLogs/osx/fink-daily-build.log
cat /private/tmp/fink-daily-build.log >> /Volumes/download/buildLogs/osx/fink-daily-build.log
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
