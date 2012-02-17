#!/bin/bash -e
#
# ./bundle-daily-build-hg.sh
#

BUNDLE="/private/tmp/orange-bundle-hg-tip.dmg"

#trap "echo \"Script failed\"" ERR

# Create the bundle
./bundle-build-hg.sh /private/tmp tip $BUNDLE

# Use local repo from the build process to get the revision
DAILY_REVISION=`hg log -r tip -R /private/tmp/repos/orange | grep 'changeset:' | cut -d ' ' -f 4 | cut -d ':' -f 1`


# TODO: Should be called only on a daily build server and not if building locally
/Users/ailabc/mount-dirs.sh

echo "Removing old versions of bundles."
# (Keeps last 5 versions.)
perl -e 'unlink ((reverse sort </Volumes/download/orange-bundle-hg-0*.dmg>)[5..10000])'

if [ -e $BUNDLE ]; then
	echo "Moving bundle to the download directory."
	mv $BUNDLE /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg
	
	echo "Setting permissions."
	chmod +r /Volumes/download/orange-bundle-hg-0.0.$DAILY_REVISION.dmg
	
	# Dont publish the bundles for now
	
	#echo "Registering new bundles."
	#egrep -v '^(MAC_STABLE|MAC_DAILY)=' /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
	#echo "MAC_STABLE=orange-bundle-1.0b.$STABLE_REVISION.dmg" >> /Volumes/download/filenames_mac.set.new
	#echo "MAC_DAILY=orange-bundle-svn-0.0.$DAILY_REVISION.dmg" >> /Volumes/download/filenames_mac.set.new
	#mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set

fi