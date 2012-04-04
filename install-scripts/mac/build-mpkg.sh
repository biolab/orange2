#!/bin/bash -ev
#
# Build mpkg package from a source distribution
#
# $1 source (.tar.gz) distribution path
# $2 work dir
# $3 distribution dir (where packages will be stored and rotated
# $4 force
#
# Example ./build-mpkg.sh http://pypi.python.org/packages/source/O/Orange/Orange-2.5a4.tar.gz /private/tmp /Volumes/download
#

SOURCE=$1
WORK_DIR=$2
DIST_DIR=$3
FORCE=$4

SOURCE_BASENAME=`basename $SOURCE`

# Could check other standard extensions (zip, tgz, ...)
DIST_NAME=${SOURCE_BASENAME%.tar.gz}

echo $DIST_NAME

if [ ! -e $WORK_DIR ]; then
	mkdir -p $WORK_DIR
fi

if [ ! -e $DIST_DIR ]; then
	mkdir -p $DIST_DIR
fi

curl --silent $SOURCE | tar -xz -C $WORK_DIR

cd $WORK_DIR/$DIST_NAME

# Use the system python
/usr/bin/python setup.py bdist_mpkg --dist-dir=$DIST_DIR

# Rotate the packages
/usr/bin/python setup.py rotate --match=.mpkg --dist-dir=$DIST_DIR --keep=20

true