#!/bin/bash
#
# Should be run as: sudo ./dailyrun.sh
#

STABLE_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
DAILY_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`

/Users/ailabc/bundle-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/bundle-daily-build.log || cat /private/tmp/bundle-daily-build.log
/Users/ailabc/fink-daily-build.sh $STABLE_REVISION $DAILY_REVISION &> /private/tmp/fink-daily-build.log || cat /private/tmp/fink-daily-build.log
