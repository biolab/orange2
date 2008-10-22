#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build.sh
#

# Those packages should not be installed as we are just building them (and dependencies)
STABLE_PACKAGES="orange-py25 orange orange-bioinformatics-py25 orange-bioinformatics orange-text-py25 orange-text"
DAILY_PACKAGES="orange-svn-py25 orange-svn orange-bioinformatics-svn-py25 orange-bioinformatics-svn orange-text-svn-py25 orange-text-svn"

# Additional source directories which get packed
SOURCE_DIRS="install-scripts/mac/bundle-lite/ add-ons/Bioinformatics/ add-ons/Text/"

# A list of packages (dependencies) from which user can choose upon installing our packages
# We would like to build all those so that it does not need to compile anything whichever packages he or she chooses
# The problem is that they are often mutually conflicting so we cannot have them simply installed (so that update-all
# would update them) but have to build them explicitly
OTHER_PACKAGES="db44 db44-aes giflib libungif ghostscript ghostscript-esp ghostscript6 ghostscript-nox ghostscript6-nox ptex-base ptex-nox-base jadetex docbook-utils tetex-base tetex-nox-base"

FINK_ARGS="--yes --build-as-nobody"
APT_ARGS="--assume-yes"

# Path to Fink root
FINK_ROOT=/sw

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

test -r $FINK_ROOT/bin/init.sh || { echo "Fink cannot be found." exit 2; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 3; }

# Configures environment for Fink
. $FINK_ROOT/bin/init.sh

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'unstable/main'; then
	echo "Fink does not seem to use unstable Fink packages tree."
	exit 4
fi

echo "Installing/updating pkgconfig package."
fink $FINK_ARGS install pkgconfig

if [ "`/usr/bin/xcodebuild -version | grep -o '^Xcode 3\.1'`" == "Xcode 3.1" ] && [ "`$FINK_ROOT/bin/pkg-config --modversion xdamage | grep -o '^[0-9]\.[0-9]'`" != "1.1" ]; then
	echo "It seems X11 version 2.3.0 or later is not installed on a system."
	exit 5
fi

echo "Preparing local ailab Fink info files repository."
mkdir -p $FINK_ROOT/fink/dists/ailab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/ailab/main/finkinfo/*

# Gets latest Fink package info files from SVN
svn export --force --non-interactive http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/ $FINK_ROOT/fink/dists/ailab/main/finkinfo/

STABLE_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
DAILY_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`

# Injects revision versions into templates
perl -pi -e "s/__STABLE_REVISION__/$STABLE_REVISION/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info
perl -pi -e "s/__DAILY_REVISION__/$DAILY_REVISION/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'ailab/main'; then
	echo "Adding local ailab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' $FINK_ROOT/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration
if ! grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' $FINK_ROOT/etc/apt/sources.list; then
	echo "Adding ailab Fink binary packages repository to Fink configuration."
	echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> $FINK_ROOT/etc/apt/sources.list
fi

mkdir -p /Volumes/fink/dists/10.5/main/source/

if [ ! -e /Volumes/fink/dists/10.5/main/source/orange-1.0b.$STABLE_REVISION.tgz ]; then
	echo "Making source archive orange-1.0b.$STABLE_REVISION."
	
	rm -rf /tmp/orange-1.0b.$STABLE_REVISION/ /tmp/orange-1.0b.$STABLE_REVISION.tgz
	
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/orange/ /tmp/orange-1.0b.$STABLE_REVISION/
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/source/ /tmp/orange-1.0b.$STABLE_REVISION/source/
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/add-ons/orngCRS/src/ /tmp/orange-1.0b.$STABLE_REVISION/source/crs/
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/COPYING /tmp/orange-1.0b.$STABLE_REVISION/COPYING
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/LICENSES /tmp/orange-1.0b.$STABLE_REVISION/LICENSES
	
	[ -e /tmp/orange-1.0b.$STABLE_REVISION/doc/COPYING ] && mv /tmp/orange-1.0b.$STABLE_REVISION/doc/COPYING /tmp/orange-1.0b.$STABLE_REVISION/
	[ -e /tmp/orange-1.0b.$STABLE_REVISION/doc/LICENSES ] && mv /tmp/orange-1.0b.$STABLE_REVISION/doc/LICENSES /tmp/orange-1.0b.$STABLE_REVISION/
	
	tar -czf /tmp/orange-1.0b.$STABLE_REVISION.tgz -C /tmp/ orange-1.0b.$STABLE_REVISION
	
	MD5SUM=`md5 -q /tmp/orange-1.0b.$STABLE_REVISION.tgz`
	perl -pi -e "s/__STABLE_MD5SUM_ORANGE__/$MD5SUM/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info
	
	mv /tmp/orange-1.0b.$STABLE_REVISION.tgz /Volumes/fink/dists/10.5/main/source/
	
	rm -rf /tmp/orange-1.0b.$STABLE_REVISION/
fi

if [ ! -e /Volumes/fink/dists/10.5/main/source/orange-svn-0.0.$DAILY_REVISION.tgz ]; then
	echo "Making source archive orange-svn-0.0.$DAILY_REVISION."
	
	rm -rf /tmp/orange-svn-0.0.$DAILY_REVISION/ /tmp/orange-svn-0.0.$DAILY_REVISION.tgz
	
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/orange/ /tmp/orange-svn-0.0.$DAILY_REVISION/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/source/ /tmp/orange-svn-0.0.$DAILY_REVISION/source/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/add-ons/orngCRS/src/ /tmp/orange-svn-0.0.$DAILY_REVISION/source/crs/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/COPYING /tmp/orange-svn-0.0.$DAILY_REVISION/COPYING
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/LICENSES /tmp/orange-svn-0.0.$DAILY_REVISION/LICENSES
	
	[ -e /tmp/orange-svn-0.0.$DAILY_REVISION/doc/COPYING ] && mv /tmp/orange-svn-0.0.$DAILY_REVISION/doc/COPYING /tmp/orange-svn-0.0.$DAILY_REVISION/
	[ -e /tmp/orange-svn-0.0.$DAILY_REVISION/doc/LICENSES ] && mv /tmp/orange-svn-0.0.$DAILY_REVISION/doc/LICENSES /tmp/orange-svn-0.0.$DAILY_REVISION/
	
	tar -czf /tmp/orange-svn-0.0.$DAILY_REVISION.tgz -C /tmp/ orange-svn-0.0.$DAILY_REVISION
	
	MD5SUM=`md5 -q /tmp/orange-svn-0.0.$DAILY_REVISION.tgz`
	perl -pi -e "s/__DAILY_MD5SUM_ORANGE__/$MD5SUM/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info
	
	mv /tmp/orange-svn-0.0.$DAILY_REVISION.tgz /Volumes/fink/dists/10.5/main/source/
	
	rm -rf /tmp/orange-svn-0.0.$DAILY_REVISION/
fi

for dir in $SOURCE_DIRS ; do
	# Gets only the last part of the directory name, converts to lower case and removes dashes
	SOURCE_NAME=`basename $dir | tr "[:upper:]" "[:lower:]" | tr -d "-"`
	STABLE_SOURCE_NAME=orange-$SOURCE_NAME-1.0b.$STABLE_REVISION
	DAILY_SOURCE_NAME=orange-$SOURCE_NAME-svn-0.0.$DAILY_REVISION
	
	if [ ! -e /Volumes/fink/dists/10.5/main/source/$STABLE_SOURCE_NAME.tgz ]; then
		echo "Making source archive $STABLE_SOURCE_NAME."
		
		rm -rf /tmp/$STABLE_SOURCE_NAME/ /tmp/$STABLE_SOURCE_NAME.tgz
		
		svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/$dir /tmp/$STABLE_SOURCE_NAME/
		svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/COPYING /tmp/$STABLE_SOURCE_NAME/COPYING
		svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/LICENSES /tmp/$STABLE_SOURCE_NAME/LICENSES
		
		[ -e /tmp/$STABLE_SOURCE_NAME/doc/COPYING ] && mv /tmp/$STABLE_SOURCE_NAME/doc/COPYING /tmp/$STABLE_SOURCE_NAME/
		[ -e /tmp/$STABLE_SOURCE_NAME/doc/LICENSES ] && mv /tmp/$STABLE_SOURCE_NAME/doc/LICENSES /tmp/$STABLE_SOURCE_NAME/
		
		tar -czf /tmp/$STABLE_SOURCE_NAME.tgz -C /tmp/ $STABLE_SOURCE_NAME
		
		MD5SUM=`md5 -q /tmp/$STABLE_SOURCE_NAME.tgz`
		perl -pi -e "s/__STABLE_MD5SUM_\U$SOURCE_NAME\E__/$MD5SUM/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info
		
		mv /tmp/$STABLE_SOURCE_NAME.tgz /Volumes/fink/dists/10.5/main/source/
	
		rm -rf /tmp/$STABLE_SOURCE_NAME/
	fi
	
	if [ ! -e /Volumes/fink/dists/10.5/main/source/$DAILY_SOURCE_NAME.tgz ]; then
		echo "Making source archive $DAILY_SOURCE_NAME."
		
		rm -rf /tmp/$DAILY_SOURCE_NAME/ /tmp/$DAILY_SOURCE_NAME.tgz
		
		svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/$dir /tmp/$DAILY_SOURCE_NAME/
		svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/COPYING /tmp/$DAILY_SOURCE_NAME/COPYING
		svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/LICENSES /tmp/$DAILY_SOURCE_NAME/LICENSES
		
		[ -e /tmp/$DAILY_SOURCE_NAME/doc/COPYING ] && mv /tmp/$DAILY_SOURCE_NAME/doc/COPYING /tmp/$DAILY_SOURCE_NAME/
		[ -e /tmp/$DAILY_SOURCE_NAME/doc/LICENSES ] && mv /tmp/$DAILY_SOURCE_NAME/doc/LICENSES /tmp/$DAILY_SOURCE_NAME/
		
		tar -czf /tmp/$DAILY_SOURCE_NAME.tgz -C /tmp/ $DAILY_SOURCE_NAME
		
		MD5SUM=`md5 -q /tmp/$DAILY_SOURCE_NAME.tgz`
		perl -pi -e "s/__DAILY_MD5SUM_\U$SOURCE_NAME\E__/$MD5SUM/g" $FINK_ROOT/fink/dists/ailab/main/finkinfo/*.info
		
		mv /tmp/$DAILY_SOURCE_NAME.tgz /Volumes/fink/dists/10.5/main/source/
	
		rm -rf /tmp/$DAILY_SOURCE_NAME/
	fi
done

# Gets all official Fink package info files
echo "Updating installed Fink packages."
fink $FINK_ARGS selfupdate --method=rsync --finish
fink $FINK_ARGS scanpackages

# Removes possiblly installed packages which we want builded
fink $FINK_ARGS purge $STABLE_PACKAGES $DAILY_PACKAGES

# Updates everything (probably by compiling new packages)
fink $FINK_ARGS update-all

# Stores current packages status
dpkg --get-selections '*' > /tmp/dpkg-selections.list

for package in $OTHER_PACKAGES ; do
	echo "Specially building package $package."
	
	# Restores intitial packages status
	dpkg --set-selections < /tmp/dpkg-selections.list
	apt-get $APT_ARGS dselect-upgrade
	
	# Builds a package if it has not been rebuilt already (for example, as a dependency)
	fink $FINK_ARGS build $package
done

# We build our packages in "maintainer" mode - Fink makes tests and validates packages
for package in $STABLE_PACKAGES $DAILY_PACKAGES ; do
	echo "Specially building, testing and validating package $package."
	
	# Restores intitial packages status
	dpkg --set-selections < /tmp/dpkg-selections.list
	apt-get $APT_ARGS dselect-upgrade
	
	# Builds a package if it has not been rebuilt already (for example, as a dependency)
	fink $FINK_ARGS --maintainer build $package
done

echo "Restoring initial packages status."
dpkg --set-selections < /tmp/dpkg-selections.list
apt-get $APT_ARGS dselect-upgrade
rm -f /tmp/dpkg-selections.list

# Cleans unncessary files (we cache them anyway in public repository)
echo "Cleaning."
fink $FINK_ARGS cleanup --all

echo "Preparing public ailab Fink info and binary files repository."
mkdir -p /Volumes/fink/dists/10.5/main/binary-darwin-i386/
mkdir -p /Volumes/fink/dists/10.5/main/finkinfo/

echo "Copying to repository all binary packages."
rsync --times --copy-links $FINK_ROOT/fink/debs/*.deb /Volumes/fink/dists/10.5/main/binary-darwin-i386/

echo "Removing old binary packages."
# (Versions of packages which have more then 5 versions and those old versions are more than one month old.)
cd /Volumes/fink/dists/10.5/main/binary-darwin-i386/
perl -e '
for (<*.deb>) {
	m/(.*?)_/;
	$fs{$1}++;
}
while (($f,$n) = each(%fs)) {
	next if $n <= 5;
	unlink for grep {-M > 30} <$f_*.deb>;
}
'

echo "Making packages list."
cd /Volumes/fink/
perl -mFink::Scanpackages -e 'Fink::Scanpackages->scan("dists/10.5/main/binary-darwin-i386/");' | gzip - > dists/10.5/main/binary-darwin-i386/Packages.gz

echo 'Archive: ailab
Origin: Fink
Component: main
Architecture: darwin-i386
Label: Fink' > dists/10.5/main/binary-darwin-i386/Release

echo "Copying to repository all info files."
rm -f /Volumes/fink/dists/10.5/main/finkinfo/*
cp $FINK_ROOT/fink/dists/ailab/main/finkinfo/* /Volumes/fink/dists/10.5/main/finkinfo/

echo "Making an archive of all info files."
cd /Volumes/fink/dists/10.5/main/finkinfo/
tar -czf all.tgz *.info
