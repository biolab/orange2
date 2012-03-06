#!/bin/bash -e
#
# Update the local/main tree with the latest packages
# from hg (needs write priviliges in FINK_ROOT/fink/dists/local/main/finkinfo
#
# Should be run as: sudo ./fink-updata-local.sh /work/dir /sources/dir
# 
# $1 Work dir (where repositories will be coned and source packages will be build)
# $2 Sources (where source packages will be copied and retrieved by fink).

if [[ ! $1 || ! $2 ]]; then
	echo "Need two args"
	exit 1
fi

WORK_DIR=$1
SOURCES_DIR=$2

# First build all the source packages
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


# Publish sources

if [ ! -e $SOURCES_DIR ]; then
	mkdir -p $SOURCES_DIR
fi

if [ ! -e $SOURCES_DIR/$ORANGE_SOURCE ]; then
	cp $WORK_DIR/$ORANGE_SOURCE $SOURCES_DIR/$ORANGE_SOURCE
	NEW_ORANGE=1
fi

if [ ! -e $SOURCES_DIR/BIOINFORMATICS_SOURCE ]; then
	cp $WORK_DIR/$BIOINFORMATICS_SOURCE $SOURCES_DIR/$BIOINFORMATICS_SOURCE
	NEW_BIOINFORMATICS=1
fi

if [ ! -e $SOURCES_DIR/TEXT_SOURCE ]; then
	cp $WORK_DIR/$TEXT_SOURCE $SOURCES_DIR/$TEXT_SOURCE
	NEW_TEXT=1
fi

FINK_ROOT=/sw

# Update the local/main finkinfo files
FINKINFO_DIR="$FINK_ROOT/fink/dists/local/main/finkinfo"

if [ ! -e $FINFINFO_DIR ]; then
	mkdir -p $FINKINFO_DIR
fi

# Directory where fink .info templates are
#FINK_TEMPLATES=fink
FINK_TEMPLATES=$WORK_DIR/orange/install-scripts/mac/fink
BASE_URL="file://$SOURCES_DIR"

if [[ $NEW_ORANGE || $FORCE ]]; then
	FINK_ORANGE_SOURCE_TEMPLATE="Orange-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-gui-hg-py.info" $BASE_URL/$FINK_ORANGE_SOURCE_TEMPLATE $ORANGE_SOURCE_MD5 $ORANGE_VERSION $FINKINFO_DIR/orange-gui-hg-py.info
fi

if [[ $NEW_BIOINFORMATICS || $FORCE ]]; then
	FINK_BIOINFORMATICS_SOURCE_TEMPLATE="Orange-Bioinformatics-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-bioinformatics-gui-hg-py.info" $BASE_URL/$FINK_BIOINFORMATICS_SOURCE_TEMPLATE $BIOINFORMATICS_SOURCE_MD5 $BIOINFORMATICS_VERSION $FINKINFO_DIR/orange-bioinformatics-gui-hg-py.info
fi

if [[ $NEW_TEXT || $FORCE ]]; then
	FINK_TEXT_SOURCE_TEMPLATE="Orange-Text-Mining-%v.tar.gz"
	./fink-register-info.sh "$FINK_TEMPLATES/orange-text-gui-hg-py.info" $BASE_URL/$FINK_TEXT_SOURCE_TEMPLATE $TEXT_SOURCE_MD5 $TEXT_VERSION $FINKINFO_DIR/orange-text-gui-hg-py.info
fi

# Index the new packages
fink index

