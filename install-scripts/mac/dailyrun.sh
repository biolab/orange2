#!/bin/bash -e
#
# Should be run as: sudo ./dailyrun.sh
#

#FORCE=true
#LOCAL=true

test -r /sw/bin/init.sh && . /sw/bin/init.sh

export PATH=$HOME/bin:$PATH

WORK_DIR=/private/tmp/repos

if [ $LOCAL ]; then
	PUBLISH_DIR=/private/tmp/download
	mkdir -p $PUBLISH_DIR
else
	PUBLISH_DIR=/Volumes/download
fi

if [ ! -e $WORK_DIR ]; then
	mkdir -p $WORK_DIR
fi


# Build source packages
./build-source.sh https://bitbucket.org/biolab/orange orange tip $WORK_DIR Orange
./build-source.sh https://bitbucket.org/biolab/orange-addon-bioinformatics bioinformatics tip $WORK_DIR Orange-Bioinformatics
./build-source.sh https://bitbucket.org/biolab/orange-addon-text text tip $WORK_DIR Orange-Text-Mining


# Get versions from PKG-INFO files
ORANGE_VERSION=`grep "^Version:" $WORK_DIR/Orange.egg-info/PKG-INFO | cut -d " " -f 2`
BIOINFORMATICS_VERSION=`grep "^Version:" $WORK_DIR/Orange_Bioinformatics.egg-info/PKG-INFO | cut -d " " -f 2`
TEXT_VERSION=`grep "^Version:" $WORK_DIR/Orange_Text_Mining.egg-info/PKG-INFO | cut -d " " -f 2`


# Source filenames
ORANGE_SOURCE="Orange-${ORANGE_VERSION}.tar.gz"
BIOINFORMATICS_SOURCE="Orange-Bioinformatics-${BIOINFORMATICS_VERSION}.tar.gz"
TEXT_SOURCE="Orange-Text-Mining-${TEXT_VERSION}.tar.gz"


# Get source packages md5 checksum
ORANGE_SOURCE_MD5=`md5 -q $WORK_DIR/$ORANGE_SOURCE`
BIOINFORMATICS_SOURCE_MD5=`md5 -q $WORK_DIR/$BIOINFORMATICS_SOURCE`
TEXT_SOURCE_MD5=`md5 -q $WORK_DIR/$TEXT_SOURCE`


MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`
ARCH=`perl -MFink::FinkVersion -e 'print Fink::FinkVersion::get_arch'`

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi

# Base url for sources
if [ $LOCAL ]; then
	BASE_URL="file://$PUBLISH_DIR/sources"
else
	BASE_URL="http://orange.biolab.si/download/sources"
fi

# Base dir for sources
SOURCES_DIR=$PUBLISH_DIR/sources


# Publish sources

if [ ! -e $SOURCES_DIR ]; then
	mkdir -p $SOURCES_DIR
fi

if [[ ! -e $SOURCES_DIR/$ORANGE_SOURCE || $FORCE]]; then
	cp $WORK_DIR/$ORANGE_SOURCE $SOURCES_DIR/$ORANGE_SOURCE
	NEW_ORANGE=1
fi

if [[ ! -e $SOURCES_DIR/BIOINFORMATICS_SOURCE || $FORCE ]]; then
	cp $WORK_DIR/$BIOINFORMATICS_SOURCE $SOURCES_DIR/$BIOINFORMATICS_SOURCE
	NEW_BIOINFORMATICS=1
fi

if [[ ! -e $SOURCES_DIR/TEXT_SOURCE || $FORCE ]]; then
	cp $WORK_DIR/$TEXT_SOURCE $SOURCES_DIR/$TEXT_SOURCE
	NEW_TEXT=1
fi

FINK_ROOT=/sw

# Update the local finkinfo 
# Local info files will be copied to biolab/main/finkinfo in fink-daily-build-packages.sh
FINK_INFO_DIR="$FINK_ROOT/fink/dists/local/main/finkinfo"

if [ ! -e $FINK_INFO_DIR ]; then
	mkdir -p $FINK_INFO_DIR
fi

# Directory where fink .info templates are
FINK_TEMPLATES=$WORK_DIR/orange/install-scripts/mac/fink

if [[ $NEW_ORANGE || $FORCE ]]; then
	FINK_ORANGE_SOURCE_TEMPLATE="Orange-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-gui-hg-py.info" $BASE_URL/$FINK_ORANGE_SOURCE_TEMPLATE $ORANGE_SOURCE_MD5 $ORANGE_VERSION $FINK_INFO_DIR/orange-gui-hg-py.info
fi

if [[ $NEW_BIOINFORMATICS || $FORCE ]]; then
	FINK_BIOINFORMATICS_SOURCE_TEMPLATE="Orange-Bioinformatics-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-bioinformatics-gui-hg-py.info" $BASE_URL/$FINK_BIOINFORMATICS_SOURCE_TEMPLATE $BIOINFORMATICS_SOURCE_MD5 $BIOINFORMATICS_VERSION $FINK_INFO_DIR/orange-bioinformatics-gui-hg-py.info
fi

if [[ $NEW_TEXT || $FORCE ]]; then
	FINK_TEXT_SOURCE_TEMPLATE="Orange-Text-Mining-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-text-gui-hg-py.info" $BASE_URL/$FINK_TEXT_SOURCE_TEMPLATE $TEXT_SOURCE_MD5 $TEXT_VERSION $FINK_INFO_DIR/orange-text-gui-hg-py.info
fi

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi


## Daily bundle build from hg
if [[ $NEW_ORANGE || $NEW_BIOINFORMATICS || $NEW_TEXT || $FORCE ]]; then
	/Users/ailabc/bundle-daily-build-hg.sh &> /private/tmp/bundle-daily-build.log
	EXIT_VALUE=$?
fi

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi

echo "Orange (bundle $MAC_VERSION from hg) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build-hg.log"
date >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build-hg.log"
cat /private/tmp/bundle-daily-build.log >> "/Volumes/download/buildLogs/osx/bundle-$MAC_VERSION-daily-build-hg.log"
(($EXIT_VALUE)) && echo "Running bundle-daily-build-hg.sh failed"


## daily fink build

/Users/ailabc/fink-daily-build-packages.sh &> /private/tmp/fink-daily-build-packages.log
EXIT_VALUE=$?

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi

echo "Orange (fink $MAC_VERSION $ARCH) [$EXIT_VALUE]" > "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
date >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
cat /private/tmp/fink-daily-build-packages.log >> "/Volumes/download/buildLogs/osx/fink-$MAC_VERSION-$ARCH-daily-build.log"
(($EXIT_VALUE)) && echo "Running fink-daily-build.sh failed"

# Zero exit value
true
