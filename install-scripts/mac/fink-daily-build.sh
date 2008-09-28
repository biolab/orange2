#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build.sh
#

STABLE_PACKAGES="orange-py25 orange orange-genomics-py25 orange-genomics"
DAILY_PACKAGES="orange-svn-py25 orange-svn orange-genomics-svn-py25 orange-genomics-svn"

# A list of packages (dependencies) from which user can choose upon installing our packages
# We would like to build all those so that it does not need to compile anything whichever packages he or she chooses
# The problem is that they are often mutually conflicting so we cannot have them simply installed (so that update-all
# would update them) but have to build them explicitly
OTHER_PACKAGES="giflib libungif ghostscript ghostscript-esp ghostscript6 ghostscript-nox ghostscript6-nox ptex-base ptex-nox-base jadetex docbook-utils tetex-base tetex-nox-base"

FINK_ARGS="--yes --build-as-nobody"

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

echo "Preparing local ailab Fink info files repository."
mkdir -p /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/*

# Gets latest Fink package info files from SVN
svn export --force --non-interactive http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/ /sw/fink/dists/ailab/main/finkinfo/

STABLE_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
DAILY_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`

# Injects revision versions into templates
perl -pi -e "s/__REVISION__/$DAILY_REVISION/g" /sw/fink/dists/ailab/main/finkinfo/*svn*.info
perl -pi -e "s/__REVISION__/$STABLE_REVISION/g" /sw/fink/dists/ailab/main/finkinfo/*.info

if ! grep '^Trees:' /sw/etc/fink.conf | grep -q 'ailab/main'; then
	echo "Adding local ailab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' /sw/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration
if ! grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' /sw/etc/apt/sources.list; then
	echo "Adding ailab Fink binary packages repository to Fink configuration."
	echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> /sw/etc/apt/sources.list
fi

# Get all official Fink package info files
echo "Updating official Fink packages."
fink $FINK_ARGS selfupdate
fink $FINK_ARGS scanpackages

# Updates everything (probably by compiling new packages)
fink $FINK_ARGS update-all

# Stores current packages status
dpkg --get-selections '*' > /tmp/dpkg-selections.list

for package in $OTHER_PACKAGES $STABLE_PACKAGES $DAILY_PACKAGES ; do
	echo "Specially building package $package."
	
	# Restores intitial packages status
	dpkg --set-selections < /tmp/dpkg-selections.list
	
	# Builds a package if it has not been rebuilt already (for example, as a dependency)
	fink $FINK_ARGS build $package
done

echo "Restoring initial packages status and cleaning."
dpkg --set-selections < /tmp/dpkg-selections.list
rm -f /tmp/dpkg-selections.list

# Cleans unncessary files
fink $FINK_ARGS cleanup --all

echo "Preparing public ailab Fink info and binary files repository."
mkdir -p /Volumes/fink/dists/10.5/main/binary-darwin-i386/
mkdir -p /Volumes/fink/dists/10.5/main/finkinfo/

echo "Copying to repository all binary packages."
rsync --times --copy-links /sw/fink/debs/*.deb /Volumes/fink/dists/10.5/main/binary-darwin-i386/

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
cp /sw/fink/dists/ailab/main/finkinfo/* /Volumes/fink/dists/10.5/main/finkinfo/

echo "Making an archive of all info files."
cd /Volumes/fink/dists/10.5/main/finkinfo/
tar -czf all.tgz *.info
