#!/bin/bash -e
#
# Should be run as: sudo ./fink-selfsync-orange.sh
#

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

# Configures environment for Fink
test -r /sw/bin/init.sh && . /sw/bin/init.sh

echo "Preparing local ailab Fink info files repository."
mkdir -p /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/*

# Gets current (daily) info files from SVN
echo "Updating local ailab Fink info files repository."
curl http://www.ailab.si/orange/fink/dists/10.5/main/finkinfo/all.tgz --output /sw/fink/dists/ailab/main/finkinfo/all.tgz
tar -xzf /sw/fink/dists/ailab/main/finkinfo/all.tgz -C /sw/fink/dists/ailab/main/finkinfo/
rm -f /sw/fink/dists/ailab/main/finkinfo/all.tgz

if ! grep '^Trees:' /sw/etc/fink.conf | grep -q 'ailab/main'; then
	echo "Adding local ailab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ ailab/main" if /^Trees/' /sw/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration
if ! grep -q 'deb http://www.ailab.si/orange/fink 10.5 main' /sw/etc/apt/sources.list; then
	echo "Adding ailab Fink binary packages repository to Fink configuration."
	echo 'deb http://www.ailab.si/orange/fink 10.5 main' >> /sw/etc/apt/sources.list
fi

# Refreshes packages lists
fink --yes scanpackages
fink --yes index

cat <<-EOF
	
	Information about ailab packages have been updated. You can now update installed
	packages using commands like 'fink update-all'.
	
	You can list available ailab packages using commands like 'fink list orange' and
	you can install them using commands like 'fink install orange'.
EOF

if ! grep '^Trees:' /sw/etc/fink.conf | grep -q 'unstable/main'; then
	cat <<-EOF
		
		WARNING: Your local Fink installation does not seem to use unstable Fink packages
		         tree. This means that it could happen that some package on which ailab
		         packages depend will not be found. In this case please configure Fink
		         to use unstable tree using 'fink configure'.
	EOF
fi