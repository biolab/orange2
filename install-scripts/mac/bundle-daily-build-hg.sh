#!/bin/bash -e
#
# ./bundle-daily-build-hg.sh
#
# $1 Force
#

FORCE=$1

trap "echo \"Script failed\"" ERR

# If possible get the orange tip revision number and check if the bundle already exists
if [ -e /private/tmp/repos/orange ]; then
	# Try to pull and update (pull returns 1 if no changesets)
	hg pull --update -R /private/tmp/repos/orange || true
	DAILY_REVISION=`hg log -r tip -R /private/tmp/repos/orange | grep 'changeset:' | cut -d ' ' -f 4 | cut -d ':' -f 1`
else
	DAILY_REVISION="tip"
fi

BUNDLE="/private/tmp/orange-bundle-hg-$DAILY_REVISION.dmg"
		
# Create the bundle if it does not yet exist
if [[ ! -e /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg || $DAILY_REVISION -eq "tip" || $FORCE ]]; then
	echo "Building orange revision $DAILY_REVISION"
	./bundle-build-hg.sh /private/tmp tip $BUNDLE
	
	# Get the revision again in case it was "tip"
	DAILY_REVISION=`hg log -r tip -R /private/tmp/repos/orange | grep 'changeset:' | cut -d ' ' -f 4 | cut -d ':' -f 1`

	# TODO: Should be called only on a daily build server and not if building locally
	/Users/ailabc/mount-dirs.sh

	echo "Removing old versions of bundles."
	# (Keeps last 5 versions.)
	perl -e 'unlink ((reverse sort </Volumes/download/orange-bundle-hg-0*.dmg>)[5..10000])'

	echo "Moving bundle to the download directory."
	mv $BUNDLE /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg

	echo "Setting permissions."
	chmod +r /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg

	# Check integrity 
	MD5=`md5 -q /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg`
	if [[ $MD5 != `md5 -q $BUNDLE` ]]; then
		echo "Error moving the bundle in place"
		rm /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg
		exit 1
	fi
	
	echo "Registering new bundles."
	egrep -v '^(MAC_DAILY)=' /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
	echo "MAC_DAILY=orange-bundle-hg-0.0.$DAILY_REVISION.dmg" >> /Volumes/download/filenames_mac.set.new
	mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set

else
	echo "The bundle with revision $DAILY_REVISION already exists."
fi