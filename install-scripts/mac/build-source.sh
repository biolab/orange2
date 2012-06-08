#!/bin/bash -e
#
# Build an Orange source package from a hg repo
#
# $1 hg repo
# $2 local repository inside work dir
# $3 hg revision
# $4 work dir
# $5 distribution name
# $6 source and egg distribution dir ($4 by default)
# $7 force
#
# Example ./build-source.sh https://bitbucket.org/biolab/orange orange tip /private/tmp/repos Orange

HG_REPO=$1
REPO_DIRNAME=$2
HG_REV=$3
WORK_DIR=$4
DIST_NAME=$5
DIST_DIR=${6:-$WORK_DIR}
FORCE=$7

# dist name with '-' replaced
UDIST_NAME=`echo $DIST_NAME | sed s/-/_/g`

LOCAL_REPO=$WORK_DIR/$REPO_DIRNAME

LATEST_REVISION=`hg id -r $HG_REV $HG_REPO`

if [ -e $DIST_DIR/$UDIST_NAME.egg-info/PKG-INFO ]; then
	CURRENT_REVISION=`grep "^Version: " $DIST_DIR/$UDIST_NAME.egg-info/PKG-INFO | cut -d "-" -f 3`
else
	CURRENT_REVISION=""
fi

if [[ $CURRENT_REVISION != $LATEST_REVISION  || $FORCE ]]; then
	BUILD=1
else
	echo "$DIST_NAME source distribution rev:$CURRENT_REVISION already exists."
	BUILD=
fi

if [ $BUILD ]; then	
	if [ ! -e $LOCAL_REPO ]; then
		hg clone $HG_REPO $LOCAL_REPO
	fi

	if hg incoming -R $LOCAL_REPO; then
		hg pull -R $LOCAL_REPO
	fi

	hg update -r $HG_REV -R $LOCAL_REPO

	hg --config extensions.purge= clean --all -R $LOCAL_REPO

	cd $LOCAL_REPO

	# Build the source distribution
	BUILD_TAG=`hg parent --template=".dev-r{rev}-{node|short}"`

	python setup.py egg_info --tag-build=$BUILD_TAG --egg-base=$DIST_DIR sdist --dist-dir=$DIST_DIR
	python setup.py rotate --match=.tar.gz --dist-dir=$DIST_DIR --keep=20
fi
