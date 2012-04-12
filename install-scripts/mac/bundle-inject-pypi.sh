#!/bin/bash -ev
#
# Install an setup.py installable package from 
# pypi
#
# $1 package name including the version (e.g. 'suds-0.4')
# $2 pypi url of the source package
# $3 work dir where the package will be downloaded and build
# $4 bundle template path
#

PACKAGE=$1
PACKAGE_URL=$2
WORK_DIR=$3
TEMPLATE_PATH=$4

# Python interpreter in the bundle 
PYTHON=${TEMPLATE_PATH}/Contents/MacOS/python

SOURCE_DIR="$WORK_DIR/$PACKAGE"
SOURCE_TAR=${SOURCE_DIR}.tar.gz

# Sets error handler
trap "echo \"Script failed\"" ERR

curl --silent -o $SOURCE_TAR $PACKAGE_URL

tar -xf $SOURCE_TAR -C $WORK_DIR

cd $SOURCE_DIR
$PYTHON setup.py install

cd ..

rm -rf $SOURCE_DIR
rm -rf $SOURCE_TAR

# 0 exit status
true