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
 

if [ ! -e $CLONE_FULLPATH ]; then
	echo "Cloning $REPO to $CLONE_FULLPATH"
	hg clone $REPO $CLONE_FULLPATH
else
	echo "Repository $CLONE_FULLPATH already present".
fi

echo "Checking for incomming changesets"
if hg incoming -R $CLONE_FULLPATH; then 
	echo "Changesets found. Pulling and updating."
	hg pull --update -R $CLONE_FULLPATH
fi

# Remove old archive if it exists
if [ -e $CLONE_ARCHIVE_FULLPATH ]; then
	echo "Removing old archive at $CLONE_ARCHIVE_FULLPATH"
	rm -rf $CLONE_ARCHIVE_FULLPATH
fi

# Create an archive
echo "Creating archive $CLONE_ARCHIVE_FULLPATH"

hg clone -r $REVISION $CLONE_FULLPATH $CLONE_ARCHIVE_FULLPATH 

cd $CLONE_ARCHIVE_FULLPATH

# Run installation
echo "Running setup.py install with python '$PYTHON'"
$PYTHON setup.py install

# Clean up the archive
cd $WORK_DIR
echo "Cleaning up the archive at $CLONE_ARCHIVE_FULLPATH"
rm -rf $CLONE_ARCHIVE_FULLPATH

true