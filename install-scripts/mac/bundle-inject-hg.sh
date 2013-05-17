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

if [ ! -e $CLONE_FULLPATH ]; then
	echo "Cloning $REPO to $CLONE_FULLPATH"
	hg clone $REPO $CLONE_FULLPATH
else
	echo "Repository $CLONE_FULLPATH already present".
fi

cd $CLONE_FULLPATH

echo "Checking for incomming changesets"
if hg incoming; then
	echo "Changesets found. Pulling."
	hg pull
fi

echo "Updating to ${REVISION}"
hg update -r ${REVISION}

# Run installation
echo "Running setup.py install with python '$PYTHON'"
$PYTHON setup.py install --single-version-externally-managed --record=RECORD.txt

true