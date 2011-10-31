#!/bin/bash -e
#
# Should be run as: sudo ./fink-selfupdate-orange.sh [/path/to/fink/root]
#

# Default is /sw
FINK_ROOT=${1:-/sw}

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

test -r $FINK_ROOT/bin/init.sh || { echo "Fink cannot be found." exit 2; }

# Configures environment for Fink
. $FINK_ROOT/bin/init.sh

if [ -e $FINK_ROOT/fink/dists/ailab/ ]; then
	echo "Removing obsolete local ailab Fink info files repository."
	rm -rf "$FINK_ROOT/fink/dists/ailab/"
fi

echo "Preparing local biolab Fink info files repository."
mkdir -p $FINK_ROOT/fink/dists/biolab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/*

# Gets current (daily) info files from SVN
echo "Updating local biolab Fink info files repository."
curl "http://orange.biolab.si/fink/dists/10.$MAC_VERSION/main/finkinfo/all.tgz" --output $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz
tar -xzf $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz -C $FINK_ROOT/fink/dists/biolab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz

if grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'ailab/main'; then
	echo "Removing obsolete local ailab Fink info files repository from Fink configuration."
	perl -p -i -l -e 's| ?ailab/main||g if /^Trees/' $FINK_ROOT/etc/fink.conf	
fi

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'biolab/main'; then
	echo "Adding local biolab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ biolab/main" if /^Trees/' $FINK_ROOT/etc/fink.conf
fi

if ! grep -q '^UseBinaryDist:.*true' $FINK_ROOT/etc/fink.conf; then
	echo "Enabling Fink binary packages repositories."
	perl -p -i -l -e 's/^UseBinaryDist:.*false/UseBinaryDist: true/' $FINK_ROOT/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration if on Mac OS X versions we provide binary packages for
if grep -q "ailab" $FINK_ROOT/etc/apt/sources.list; then
	echo "Removing obsolete ailab Fink binary packages repository from Fink configuration."
	grep -v "ailab" $FINK_ROOT/etc/apt/sources.list > $FINK_ROOT/etc/apt/sources.list.new
	mv $FINK_ROOT/etc/apt/sources.list.new $FINK_ROOT/etc/apt/sources.list
	apt-get update
fi

if [[ "$MAC_VERSION" -eq 5 || "$MAC_VERSION" -eq 6 ]] && ! grep -q "deb http://orange.biolab.si/fink 10.$MAC_VERSION main" $FINK_ROOT/etc/apt/sources.list; then
	echo "Adding biolab Fink binary packages repository to Fink configuration."
	echo "deb http://orange.biolab.si/fink 10.$MAC_VERSION main" >> $FINK_ROOT/etc/apt/sources.list
fi

# Refreshes packages lists
fink --yes scanpackages
fink --yes index

cat <<-EOMSG
	
	Information about biolab packages have been updated. You can now update installed
	packages using:
	
	    fink update-all
	
	You can list available biolab packages using commands like:
	
	    fink list --tab orange
	
	and you can install them using commands like:
	
	    fink install orange-gui-svn-py27
EOMSG

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'unstable/main' || grep '^SelfUpdateMethod:' $FINK_ROOT/etc/fink.conf | grep -q 'point'; then
	cat <<-EOMSG
		
		WARNING: Your local Fink installation does not seem to use unstable Fink
		         packages tree with rsync or CVS updating. This means that it could
		         happen that some package on which biolab packages depend will not be
		         found or possible to install. In this case please configure Fink to
		         use unstable tree using:
		         
		             fink configure
		         
		         and after that upgrade Fink to rsync updating using:
		         
		             fink scanpackages
		             fink selfupdate
		             fink selfupdate-rsync
		             fink update-all
	EOMSG
	
	if [ ! -x /usr/bin/xcodebuild ]; then
		cat <<-EOMSG
			         
			         You will need Xcode installed on a system for above commands to work
			         correctly as it is a requirement for unstable Fink packages tree. You
			         can install it from your Mac OS X installation disk or from:
			         
			             http://developer.apple.com/technology/xcode.html
		EOMSG
	fi
fi

if [ $MAC_VERSION -lt "5" ] && [ ! -x /usr/X11/bin/X ]; then
	cat <<-EOMSG
		
		WARNING: It seems you do not have X11 installed on a system. This means that it
		         could happen that some package on which biolab packages depend will fail
		         to run or compile. In this case please install it from your Mac OS X
		         installation disk.
	EOMSG
elif [ $MAC_VERSION -ge "5" ] && [ ! "`/usr/X11/bin/X -version 2>&1 | grep '^X.Org X Server' | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+' | cut -d '.' -f 2`" -gt "3" ]; then
	cat <<-EOMSG
		
		WARNING: It seems you do not have X11 version 2.3.0 or later installed on a
		         system. This means that it could happen that some package on which
		         biolab packages depend will fail to run or compile. In this case
		         please install it from your Mac OS X installation disk or from:
		         
		             http://xquartz.macosforge.org/trac/wiki
	EOMSG
fi
