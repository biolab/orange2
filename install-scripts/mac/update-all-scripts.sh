#!/bin/bash
#
# Should be run as: ./update-all-scripts.sh
#

curl --silent --output update-all-scripts.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/update-all-scripts.sh
curl --silent --output bundle-64bit-daily-build.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-64bit-daily-build.sh
curl --silent --output bundle-daily-build.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-daily-build.sh
curl --silent --output dailyrun.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun.sh
#curl --silent --output dailyrun-finkonly-withsource.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun-finkonly-withsource.sh
#curl --silent --output dailyrun-finkonly.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun-finkonly.sh
curl --silent --output dailyrun-bundleonly.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun-bundleonly.sh
#curl --silent --output fink-daily-build.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/fink-daily-build.sh
#curl --silent --output fink-restore-selections.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/fink-restore-selections.sh
#curl --silent --output fink-selfupdate-orange.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/fink-selfupdate-orange.sh
#curl --silent --output force-fink-daily-build.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/force-fink-daily-build.sh
curl --silent --output mount-dirs.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/mount-dirs.sh

curl --silent --output bundle-build-hg.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-build-hg.sh
curl --silent --output bundle-daily-build-hg.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-daily-build-hg.sh
curl --silent --output bundle-inject-hg.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-inject-hg.sh
curl --silent --output bundle-inject-pypi.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/bundle-inject-pypi.sh
curl --silent --output dailyrun-bundleonly-hg.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun-bundleonly-hg.sh
#curl --silent --output fink-daily-build-packages.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/fink-daily-build-packages.sh
#curl --silent --output fink-register-info.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/fink-register-info.sh
curl --silent --output build-source.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/build-source.sh
curl --silent --output dailyrun-sources.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/dailyrun-sources.sh
curl --silent --output build-mpkg.sh https://bitbucket.org/biolab/orange/raw/tip/install-scripts/mac/build-mpkg.sh

chmod +x *.sh

# Zero exit value
true
