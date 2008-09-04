#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build-svn.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

# Gets latest Fink package info files from SVN to local info files repository 
cd /sw/fink/dists/local/main/
svn export http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/
mv fink/* finkinfo/
rm -rf fink/

# Get all official Fink package info files
fink selfupdate
fink scanpackages

# Updates everything (probably by compiling new packages)
fink update-all

# Copies to repository all binary packages
# (Maybe it would be better to rsync them? This also keeps obsolete and maybe non-existent packages in repository.)
cp /sw/fink/debs/*.deb /Volumes/fink/dists/10.5/main/binary-darwin-i386/

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
