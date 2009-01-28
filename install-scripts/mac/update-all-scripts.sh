#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

curl --silent --output update-all-scripts.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/update-all-scripts.sh
curl --silent --output bundle-64bit-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/bundle-64bit-daily-build.sh
curl --silent --output bundle-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/bundle-daily-build.sh
curl --silent --output dailyrun.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/dailyrun.sh
curl --silent --output dailyrun-finkonly.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/dailyrun-finkonly.sh
curl --silent --output fink-daily-build.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-daily-build.sh
curl --silent --output fink-restore-selections.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-restore-selections.sh
curl --silent --output fink-selfupdate-orange.sh http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-selfupdate-orange.sh

chmod +x *.sh
