#!/bin/bash
#
# Should be run as: sudo ./dailyrun.sh
#

STABLE_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ $STABLE_REVISION ] || exit 1

DAILY_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ $DAILY_REVISION ] || exit 1

echo "Running bundle-daily-build.sh" > /private/tmp/bundle-daily-build.log
/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log || cat /private/tmp/bundle-daily-build.log

echo "Running fink-daily-build.sh" > /private/tmp/bundle-daily-build.log
/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/fink-daily-build.log || cat /private/tmp/fink-daily-build.log
