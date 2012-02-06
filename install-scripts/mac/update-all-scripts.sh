#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

# Clone the orange repo if not already present
if [ ! -e orange ]; then
	hg clone https://bitbucket.org/biolab/orange
fi

# Pull all changesets and update to latest
cd orange
hg pull --update
cd ..

cp orange/install-scripts/mac/update-all-scripts.sh ./
cp orange/install-scripts/mac/bundle-64bit-daily-build.sh ./
cp orange/install-scripts/mac/bundle-daily-build.sh ./
cp orange/install-scripts/mac/dailyrun.sh ./
cp orange/install-scripts/mac/dailyrun-finkonly-withsource.sh ./
cp orange/install-scripts/mac/dailyrun-finkonly.sh ./
cp orange/install-scripts/mac/dailyrun-bundleonly.sh ./
cp orange/install-scripts/mac/fink-daily-build.sh ./
cp orange/install-scripts/mac/fink-restore-selections.sh ./
cp orange/install-scripts/mac/fink-selfupdate-orange.sh ./
cp orange/install-scripts/mac/force-fink-daily-build.sh ./
cp orange/install-scripts/mac/mount-dirs.sh ./

chmod +x *.sh

# Zero exit value
true
