#!/bin/bash -e
#
# Should be run as: sudo ./fink-instal-orange-svn.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

# Gets latest Fink package info files from SVN to local info files repository 
cd /sw/fink/dists/local/main/
svn export http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink/
mv fink/* finkinfo/
rm -rf fink/

# Get all official Fink package info files (as maybe new our Fink packages depends on them)
fink selfupdate

# Installs packages
fink install orange-svn-py25 orange-svn orange-genomics-svn-py25 orange-genomics-svn
