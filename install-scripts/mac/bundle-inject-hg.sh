#!/bin/bash -e
#
# Install an setup.py installable package from a 
# hg repository in to a .app bundle template
#
# $1 hg repo path
# $2 repo clone name inside the working dir
# $3 revision
# $4 working directory where the repos will be cloned
# $5 bundle template path
#

REPO=$1
CLONE_NAME=$2
REVISION=$3
WORK_DIR=$4
TEMPLATE_PATH=$5

# Python interpreter in the bundle 
PYTHON=${TEMPLATE_PATH}/Contents/MacOS/python

# Sets error handler
trap "echo \"Script failed\"" ERR

# Path to the local repo clone
CLONE_FULLPATH=${WORK_DIR}/${CLONE_NAME}

# Path to the local archived source. This is where the building 
# will actually take place to prevent the polution of the repo
CLONE_ARCHIVE_NAME=${CLONE_NAME}_archive
CLONE_ARCHIVE_FULLPATH=${WORK_DIR}/${CLONE_ARCHIVE_NAME}
 
# If the repo clone does not yet exist then clone it
if [ ! -e $CLONE_FULLPATH ]; then
	hg clone $REPO $CLONE_FULLPATH
fi

hg pull --update -R $CLONE_FULLPATH

# Remove old archive if it exists
if [ -e $CLONE_ARCHIVE_FULLPATH ]; then
	rm -rf $CLONE_ARCHIVE_FULLPATH
fi

# Create an archive
hg archive -r $REVISION $CLONE_ARCHIVE_FULLPATH -R $CLONE_FULLPATH

cd $CLONE_ARCHIVE_FULLPATH

# Run installation
$PYTHON setup.py install

# Clean up the archive
cd $WORK_DIR
rm -rf $CLONE_ARCHIVE_FULLPATH

true