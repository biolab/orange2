#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

# Prepares our Fink package info files repository
mkdir -p /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/*

# Gets latest Fink package info files from SVN to our info files repository 
svn export --force --non-interactive http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/ /sw/fink/dists/ailab/main/finkinfo/

STABLE_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`
TRUNK_REVISION=`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`

# Inject revision versions into templates
perl -pi -e "s/__REVISION__/$TRUNK_REVISION/g" /sw/fink/dists/ailab/main/finkinfo/*svn*.info
perl -pi -e "s/__REVISION__/$STABLE_REVISION/g" /sw/fink/dists/ailab/main/finkinfo/*.info

# Adds our info files repository to local Fink configuration
grep '^Trees:' /sw/etc/fink.conf | grep -q 'ailab/main' || perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' /sw/etc/fink.conf

# Adds our binary repository to local Fink (APT) configuration (so it does not rebuild packages unnecessarily if local copies were removed)
grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' /sw/etc/apt/sources.list || echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> /sw/etc/apt/sources.list

# Get all official Fink package info files
fink --yes selfupdate
fink --yes scanpackages

# Updates everything (probably by compiling new packages)
fink --yes update-all

# Prepares our public Fink package info and binary files repository
mkdir -p /Volumes/fink/dists/10.5/main/binary-darwin-i386/
mkdir -p /Volumes/fink/dists/10.5/main/finkinfo/

# Copies to repository all binary packages
rsync -t /sw/fink/debs/*.deb /Volumes/fink/dists/10.5/main/binary-darwin-i386/

# Removes old binary packages
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

# Makes new packages list
cd /Volumes/fink/
perl -mFink::Scanpackages -e 'Fink::Scanpackages->scan("dists/10.5/main/binary-darwin-i386/");' | gzip - > dists/10.5/main/binary-darwin-i386/Packages.gz

echo 'Archive: ailab
Origin: Fink
Component: main
Architecture: darwin-i386
Label: Fink' > dists/10.5/main/binary-darwin-i386/Release

# Copies to repository all info files
rm -f /Volumes/fink/dists/10.5/main/finkinfo/*
cp /sw/fink/dists/ailab/main/finkinfo/* /Volumes/fink/dists/10.5/main/finkinfo/
