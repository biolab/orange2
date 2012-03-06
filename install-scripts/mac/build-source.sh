#!/bin/bash -e
#
# Build an Orange source package from a hg repo
#
# $1 hg repo
# $2 local repository inside work dir
# $3 hg revision
# $4 work dir
# $5 distribution name
#
# Example ./build-source.sh http://bitbucket.org/biolab/orange orange tip /private/tmp/repos Orange

HG_REPO=$1
REPO_DIRNAME=$2
HG_REV=$3
WORK_DIR=$4
DIST_NAME=$5

LOCAL_REPO=$WORK_DIR/$REPO_DIRNAME

if [ ! -e $LOCAL_REPO ]; then
	hg clone $HG_REPO $LOCAL_REPO
fi

if hg incoming -R $LOCAL_REPO; then
	hg pull -R $LOCAL_REPO
fi

hg update -r $HG_REV -R $LOCAL_REPO

cd $LOCAL_REPO

# Remove old sources
rm -rf dist/

# Build the source distribution
hg parent --template="{latesttag}{latesttagdistance}.dev-{node|short}" > VERSION.txt
python setup.py sdist

# Copy the source an egg info to workdir
cp dist/${DIST_NAME}-*.tar.gz $WORK_DIR

UDIST_NAME=`echo $DIST_NAME | sed s/-/_/g`
cp -r  $UDIST_NAME.egg-info $WORK_DIR

true