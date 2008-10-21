#!/bin/bash -e
#
# Should be run as: sudo ./fink-selfsync-orange.sh /path/to/fink/root
#

# Default is /sw
FINK_ROOT=${1:-/sw}

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

test -r $FINK_ROOT/bin/init.sh || { echo "Fink cannot be found." exit 2; }

# Configures environment for Fink
. $FINK_ROOT/bin/init.sh

echo "Preparing local ailab Fink info files repository."
mkdir -p $FINK_ROOT/fink/dists/ailab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/ailab/main/finkinfo/*

# Gets current (daily) info files from SVN
echo "Updating local ailab Fink info files repository."
curl http://www.ailab.si/orange/fink/dists/10.5/main/finkinfo/all.tgz --output $FINK_ROOT/fink/dists/ailab/main/finkinfo/all.tgz
tar -xzf $FINK_ROOT/fink/dists/ailab/main/finkinfo/all.tgz -C $FINK_ROOT/fink/dists/ailab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/ailab/main/finkinfo/all.tgz

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'ailab/main'; then
	echo "Adding local ailab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' $FINK_ROOT/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration
if ! grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' $FINK_ROOT/etc/apt/sources.list; then
	echo "Adding ailab Fink binary packages repository to Fink configuration."
	echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> $FINK_ROOT/etc/apt/sources.list
fi

# Refreshes packages lists
fink --yes scanpackages
fink --yes index

echo "Installing/updating pkgconfig package."
fink install pkgconfig

cat <<-EOMSG
	
	Information about ailab packages have been updated. You can now update installed
	packages using:
	
	    fink update-all
	
	You can list available ailab packages using commands like:
	
	    fink list --tab orange
	
	and you can install them using commands like:
	
	    fink install orange
EOMSG

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'unstable/main'; then
	cat <<-EOMSG
		
		WARNING: Your local Fink installation does not seem to use unstable Fink
		         packages tree. This means that it could happen that some package on
		         which ailab packages depend will not be found. In this case please
		         configure Fink to use unstable tree using:
		         
		             fink configure
		         
		         and after that upgrade Fink using:
		         
		             fink selfupdate-rsync
	EOMSG
fi

if [ "`/usr/bin/xcodebuild -version | grep -o '^Xcode 3\.1'`" == "Xcode 3.1" ] && [ "`$FINK_ROOT/bin/pkg-config --modversion xdamage | grep -o '^[0-9]\.[0-9]'`" != "1.1" ]; then
	cat <<-EOMSG
		
		WARNING: It seems you do not have X11 version 2.3.0 or later installed on a
		         system. This means that it could happen that some package on which
		         ailab packages depend will fail to run or compile. In this case
		         please install newer version from:
		         
		             http://xquartz.macosforge.org/trac/
	EOMSG
fi
