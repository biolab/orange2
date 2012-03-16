#!/bin/bash -e
#
# Should be run as: sudo ./dailyrun-sources.sh work/dir
#
# $1 work dir
# $2 force
# $3 local build

WORK_DIR=${1:-"/private/tmp"}

FORCE=$2
LOCAL=$3

test -r /sw/bin/init.sh && . /sw/bin/init.sh

export PATH=$HOME/bin:$PATH

if [ $LOCAL ]; then
	PUBLISH_DIR=$WORK_DIR/download
	mkdir -p $PUBLISH_DIR
else
	PUBLISH_DIR=/Volumes/download
fi

if [ ! -e $WORK_DIR ]; then
	mkdir -p $WORK_DIR
fi

REPO_DIR=$WORK_DIR/repos

# Build source packages
./build-source.sh https://bitbucket.org/biolab/orange orange tip $REPO_DIR Orange
./build-source.sh https://bitbucket.org/biolab/orange-addon-bioinformatics bioinformatics tip $REPO_DIR Orange-Bioinformatics
./build-source.sh https://bitbucket.org/biolab/orange-addon-text text tip $REPO_DIR Orange-Text-Mining

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi

# Get versions from PKG-INFO files
ORANGE_VERSION=`grep "^Version:" $REPO_DIR/Orange.egg-info/PKG-INFO | cut -d " " -f 2`
BIOINFORMATICS_VERSION=`grep "^Version:" $REPO_DIR/Orange_Bioinformatics.egg-info/PKG-INFO | cut -d " " -f 2`
TEXT_VERSION=`grep "^Version:" $REPO_DIR/Orange_Text_Mining.egg-info/PKG-INFO | cut -d " " -f 2`

# Source filenames
ORANGE_SOURCE="Orange-${ORANGE_VERSION}.tar.gz"
BIOINFORMATICS_SOURCE="Orange-Bioinformatics-${BIOINFORMATICS_VERSION}.tar.gz"
TEXT_SOURCE="Orange-Text-Mining-${TEXT_VERSION}.tar.gz"

# Get source packages md5 checksum
ORANGE_SOURCE_MD5=`md5 -q $REPO_DIR/$ORANGE_SOURCE`
BIOINFORMATICS_SOURCE_MD5=`md5 -q $REPO_DIR/$BIOINFORMATICS_SOURCE`
TEXT_SOURCE_MD5=`md5 -q $REPO_DIR/$TEXT_SOURCE`


# Base dir for sources
SOURCES_DIR=$PUBLISH_DIR/sources


# Publish sources

if [ ! -e $SOURCES_DIR ]; then
	mkdir -p $SOURCES_DIR
fi


if [[ ! -e $SOURCES_DIR/$ORANGE_SOURCE || $FORCE ]]; then
	cp $REPO_DIR/$ORANGE_SOURCE $SOURCES_DIR/$ORANGE_SOURCE
	cp -r $REPO_DIR/Orange.egg-info $SOURCES_DIR/
	NEW_ORANGE=1
fi

if [[ ! -e $SOURCES_DIR/BIOINFORMATICS_SOURCE || $FORCE ]]; then
	cp $REPO_DIR/$BIOINFORMATICS_SOURCE $SOURCES_DIR/$BIOINFORMATICS_SOURCE
	cp -r $REPO_DIR/Orange_Bioinformatics.egg-info $SOURCES_DIR/
	NEW_BIOINFORMATICS=1
fi

if [[ ! -e $SOURCES_DIR/TEXT_SOURCE || $FORCE ]]; then
	cp $REPO_DIR/$TEXT_SOURCE $SOURCES_DIR/$TEXT_SOURCE
	cp -r $REPO_DIR/Orange_Text_Mining.egg-info $SOURCES_DIR/
	NEW_TEXT=1
fi

true
