#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

wget --quiet --output-document=update-all-scripts.sh http://orange.biolab.si/svn/orange/trunk/install-scripts/debian/update-all-scripts.sh
wget --quiet --output-document=debian-daily-build.sh http://orange.biolab.si/svn/orange/trunk/install-scripts/debian/debian-daily-build.sh
wget --quiet --output-document=dailyrun.sh http://orange.biolab.si/svn/orange/trunk/install-scripts/debian/dailyrun.sh

chmod +x *.sh

# Zero exit value
true
