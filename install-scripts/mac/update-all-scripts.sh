#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

cd /Users/ailabc/

curl -o update-all-scripts.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/update-all-scripts.sh
curl -o bundle-64bit-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/bundle-64bit-daily-build.sh
curl -o bundle-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/bundle-daily-build.sh
curl -o dailyrun.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/dailyrun.sh
curl -o fink-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-daily-build.sh
curl -o fink-restore-selections.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-restore-selections.sh
curl -o fink-selfupdate-orange.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-selfupdate-orange.sh

chmod +x *.sh
