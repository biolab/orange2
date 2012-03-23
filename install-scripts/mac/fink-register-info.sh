#!/bin/bash -e
#
# Register new revision of the fink package
#
# $1 fink-info file template
# $2 source package url
# $3 source package md5
# $4 fink package version
# $5 final .info file (optional /sw/fink/dists/local/main/finkinfo/$1 will be used
#
# Example: ./register-fink-info fink/orange-gui-dev-py.info http://orange.biolab.si/downloads/sources/Orange-2.5a.tar.gz 2fa4783166c07585ae349723495cf2b8 2.5a /sw/fink/dists/local/main/finkinfo/orange-gui-dev-py.info

INFO=$1
SOURCE=$2
MD5=$3
VERSION=$4

if [ $5 ]; then
	echo "fink: $5"
	PUBLISH=$5
else
	PUBLISH="/sw/fink/dists/local/main/finkinfo/$1"
fi

echo $PUBLISH

# Escape the / characters in source
SOURCE_ESCAPED=`echo $SOURCE | sed s/'\/'/'\\\\\/'/g`


cat $INFO | sed s/__SOURCE__/$SOURCE_ESCAPED/g | \
	sed s/__VERSION__/$VERSION/g | sed s/__MD5SUM__/$MD5/g > $PUBLISH

true