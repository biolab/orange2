#!/bin/bash -e
#
# Should be run as: sudo ./fink-selfsync-orange.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

# Prepares our Fink package info files repository
mkdir -p /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/*

# Gets current (daily) Fink package info files from SVN to our info files repository 
curl http://www.ailab.si/orange/fink/dists/10.5/main/finkinfo/all.tgz --output /sw/fink/dists/ailab/main/finkinfo/all.tgz
tar -xzf /sw/fink/dists/ailab/main/finkinfo/all.tgz -C /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/all.tgz

# Adds our info files repository to local Fink configuration
grep '^Trees:' /sw/etc/fink.conf | grep -q 'ailab/main' || perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' /sw/etc/fink.conf

# Adds our binary repository to local Fink (APT) configuration (so it does not rebuild packages unnecessarily if local copies were removed)
grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' /sw/etc/apt/sources.list || echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> /sw/etc/apt/sources.list

fink --all scanpackages
