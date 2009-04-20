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

[ -e /Volumes/fink/ ] || { open "smb://orange@estelle.fri.uni-lj.si/fink/"; sleep 30; }
[ -e /Volumes/download/ ] || { open "smb://orange@estelle.fri.uni-lj.si/download/"; sleep 30; }

echo "Running bundle-daily-build.sh" > /private/tmp/bundle-daily-build.log
/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log || cat /private/tmp/bundle-daily-build.log

echo "Running bundle-64bit-daily-build.sh" > /private/tmp/bundle-64bit-daily-build.log
/Users/ailabc/bundle-64bit-daily-build.sh $DAILY_REVISION &> /private/tmp/bundle-64bit-daily-build.log || cat /private/tmp/bundle-64bit-daily-build.log

echo "Running fink-daily-build.sh" > /private/tmp/fink-daily-build.log
/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/fink-daily-build.log || cat /private/tmp/fink-daily-build.log
