#!/bin/bash
#
# Should be run as: sudo ./dailyrun.sh
#

DAILY_REVISION=`svn info --non-interactive http://orange.biolab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION" ] || exit 1

ARCH=`dpkg --print-architecture`

mount | grep -q /mnt/debian || { mount /mnt/debian; sleep 10; }
mount | grep -q /mnt/download || { mount /mnt/download; sleep 10; }

/root/debian-daily-build.sh $DAILY_REVISION &> /tmp/debian-daily-build.log
EXIT_VALUE=$?
echo "Orange (Debian $ARCH) [$EXIT_VALUE]" > "/mnt/download/buildLogs/debian/debian-$ARCH-daily-build.log"
date >> "/mnt/download/buildLogs/debian/debian-$ARCH-daily-build.log"
cat /tmp/debian-daily-build.log >> "/mnt/download/buildLogs/debian/debian-$ARCH-daily-build.log"
(($EXIT_VALUE)) && echo "Running debian-daily-build.sh ($ARCH) failed"

# Zero exit value
true
