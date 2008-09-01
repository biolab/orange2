#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build-svn.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

# Gets latest Fink package info files from SVN to local info files repository 
cd /sw/fink/dists/local/main/
svn export http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/
mv fink/* finkinfo/
rm -rf fink/

fink scanpackages
fink selfupdate

fink update-all

cp /sw/fink/debs/*.deb /Volumes/fink/10.5/svn/

cd /Volumes/fink/10.5/
dpkg-scanpackages svn /dev/null | gzip - > svn/Packages.gz
