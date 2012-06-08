#!/bin/bash -e
#
# Should be run as: sudo ./dailyrun-sources.sh work/dir
#
# $1 work dir
# $2 local build
# $3 force

WORK_DIR=${1:-"/private/tmp"}

LOCAL=$2
FORCE=$3

test -r /sw/bin/init.sh && . /sw/bin/init.sh

export PATH=$HOME/bin:$PATH

if [ $LOCAL ]; then
	PUBLISH_DIR=$WORK_DIR/download/sources
else
	PUBLISH_DIR=/Volumes/download/sources
fi

if [ ! -e $WORK_DIR ]; then
	mkdir -p $WORK_DIR
fi


REPO_DIR=$WORK_DIR/repos

defaults write com.apple.desktopservices DSDontWriteNetworkStores true

if [ ! $LOCAL ]; then
	/Users/ailabc/mount-dirs.sh || { echo "Mounting failed." ; exit 1 ; }
fi

if [ ! -e $PUBLISH_DIR ]; then
	mkdir -p $PUBLISH_DIR
fi

# Build source packages
./build-source.sh https://bitbucket.org/biolab/orange orange tip $REPO_DIR Orange $PUBLISH_DIR $FORCE
./build-source.sh https://bitbucket.org/biolab/orange-bioinformatics bioinformatics tip $REPO_DIR Orange-Bioinformatics $PUBLISH_DIR $FORCE
./build-source.sh https://bitbucket.org/biolab/orange-text text tip $REPO_DIR Orange-Text $PUBLISH_DIR $FORCE

true