#!/bin/bash -e
#
# ./bundle-daily-build-hg.sh
#
# $1 workdir (directory where sources can be checked out and build)
# $2 publishdir (directory where the resulting versioned bundle can be moved)
# $3 force (force the build even if a bundle with the same revision exists at publishdir)
# $4 local
#

WORK_DIR=${1:-"/private/tmp"}
PUBLISH_DIR=${2:-"$WORK_DIR/download"}

FORCE=$3
LOCAL=$4

ORANGE_REPO=$WORK_DIR/repos/orange

trap "echo \"Script failed\"" ERR

# If possible get the orange tip revision number and check if the bundle already exists
if [ -e $ORANGE_REPO ]; then
	# Try to pull and update (pull returns 1 if no changesets)
	hg pull --update -R $ORANGE_REPO || true
	DAILY_REVISION=`hg log -r tip -R $ORANGE_REPO | grep 'changeset:' | cut -d ' ' -f 4 | cut -d ':' -f 1`
else
	DAILY_REVISION="tip"
fi

BUNDLE="$WORK_DIR/orange-bundle-hg-$DAILY_REVISION.dmg"
PUBLISH_BUNDLE=$PUBLISH_DIR/orange-bundle-hg-0.0.$DAILY_REVISION.dmg


# Create the bundle if it does not yet exist
if [[ ! -e $PUBLISH_BUNDLE || $DAILY_REVISION -eq "tip" || $FORCE ]]; then
	echo "Building orange revision $DAILY_REVISION"
	./bundle-build-hg.sh $WORK_DIR tip $BUNDLE
	
	# Get the revision again in case it was "tip"
	DAILY_REVISION=`hg log -r tip -R $ORANGE_REPO | grep 'changeset:' | cut -d ' ' -f 4 | cut -d ':' -f 1`
	# And update the publish bundle filename
	PUBLISH_BUNDLE=$PUBLISH_DIR/orange-bundle-hg-0.0.$DAILY_REVISION.dmg

	if [ ! $LOCAL ]; then
		/Users/ailabc/mount-dirs.sh
	fi

	echo "Removing old versions of bundles."
	# (Keeps last 5 versions.)
	perl -e "unlink ((reverse sort <$PUBLISH_DIR/orange-bundle-hg-0*.dmg>)[5..10000])"

	MD5=`md5 -q $BUNDLE`
	
	echo "Moving bundle to the download directory."
	mv $BUNDLE $PUBLISH_BUNDLE.new

	echo "Setting permissions."
	chmod +r $PUBLISH_BUNDLE.new

	# Check integrity
	MD5_D=`md5 -q $PUBLISH_BUNDLE.new`
	if [[ $MD5 != $MD5_D ]]; then
		echo "Error moving the bundle in place"
		rm $PUBLISH_BUNDLE.new
		exit 1
	else
		mv $PUBLISH_BUNDLE.new $PUBLISH_BUNDLE
	fi
	
	echo "Registering new bundles."
	egrep -v '^(MAC_DAILY)=' $PUBLISH_DIR/filenames_mac.set > $PUBLISH_DIR/filenames_mac.set.new
	echo "MAC_DAILY=orange-bundle-hg-0.0.$DAILY_REVISION.dmg" >> $PUBLISH_DIR/filenames_mac.set.new
	mv $PUBLISH_DIR/filenames_mac.set.new $PUBLISH_DIR/filenames_mac.set

else
	echo "The bundle with revision $DAILY_REVISION already exists."
fi