#!/bin/bash -e
#
# Should be run as: sudo ./debian-daily-build.sh [daily revision]
#

APT_ARGS="--assume-yes"
APTITUDE_ARGS="--assume-yes"
BUILD_DIR="/tmp/orange-build"

ARCH=`dpkg --print-architecture`
DISTRIBUTION=`lsb_release -c -s`

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

[ -e /mnt/debian/ ] || { echo "/mnt/debian/ not mounted."; exit 2; }

# Default is current latest revision in trunk
DAILY_REVISION=${1:-`svn info --non-interactive http://orange.biolab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION" ] || exit 3

# Adds our repository to APT configuration
if ! grep -q "deb http://orange.biolab.si/debian $DISTRIBUTION main" /etc/apt/sources.list; then
	echo "Adding biolab packages repository to APT configuration."
	echo "deb http://orange.biolab.si/debian $DISTRIBUTION main" >> /etc/apt/sources.list
	echo "deb-src http://orange.biolab.si/debian $DISTRIBUTION main" >> /etc/apt/sources.list
fi

# We are checking only for -1 debian revision as those are those made by uupdate
if [ -e "/mnt/debian/dists/$DISTRIBUTION/main/binary-$ARCH/python-orange_0.0.$DAILY_REVISION~svn-1_$ARCH.deb" ]; then
	echo "Package for $DAILY_REVISION revision already exists."
	exit 0
fi

aptitude $APTITUDE_ARGS update
aptitude $APTITUDE_ARGS install devscripts build-essential
aptitude $APTITUDE_ARGS build-depends python-orange

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

apt-get $APT_ARGS source python-orange

if [ -e "python-orange-0.0.$DAILY_REVISION~svn" ]; then
	echo "Package for $DAILY_REVISION revision already exists, just building it."
else
	echo "Making source archive python-orange-0.0.$DAILY_REVISION~svn."
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/orange/ python-orange-0.0.$DAILY_REVISION~svn/
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/source/ python-orange-0.0.$DAILY_REVISION~svn/source/
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/add-ons/orngCRS/src/ python-orange-0.0.$DAILY_REVISION~svn/source/crs/
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/COPYING python-orange-0.0.$DAILY_REVISION~svn/COPYING
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/LICENSES python-orange-0.0.$DAILY_REVISION~svn/LICENSES
	
	[ -e python-orange-0.0.$DAILY_REVISION~svn/doc/COPYING ] && mv python-orange-0.0.$DAILY_REVISION~svn/doc/COPYING python-orange-0.0.$DAILY_REVISION~svn/
	[ -e python-orange-0.0.$DAILY_REVISION~svn/doc/LICENSES ] && mv python-orange-0.0.$DAILY_REVISION~svn/doc/LICENSES python-orange-0.0.$DAILY_REVISION~svn/
	
	tar -czf python-orange-0.0.$DAILY_REVISION~svn.tar.gz python-orange-0.0.$DAILY_REVISION~svn
	rm -rf python-orange-0.0.$DAILY_REVISION~svn/
	
	echo "Updating packages."
	cd python-orange-0.0.*/
	export DEBFULLNAME="Mitar"
	export DEBEMAIL="mitar@tnode.com"
	uupdate --upstream-version 0.0.$DAILY_REVISION~svn --no-symlink python-orange-0.0.$DAILY_REVISION~svn.tar.gz
	cd ..
	rm -rf python-orange-0.0.$DAILY_REVISION~svn.orig/ python-orange-0.0.$DAILY_REVISION~svn.tar.gz
	
	echo "Updating Debian packaging files."
	cd python-orange-0.0.$DAILY_REVISION~svn/debian/
	svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/install-scripts/debian/control-files
	cd control-files
	rm -f changelog
	mv -f * ../
	cd ../
	rm -rf control-files
	cd ../../
	
	echo "Building new packages."
fi

cd python-orange-0.0.$DAILY_REVISION~svn/
dpkg-buildpackage -D -sa -us -uc

echo "Preparing public biolab Debian repository."
mkdir -p /mnt/debian/dists/$DISTRIBUTION/main/binary-$ARCH/
mkdir -p /mnt/debian/dists/$DISTRIBUTION/main/source/

echo "Copying to repository new packages."
cd ..
rm -rf python-orange-0.0.$DAILY_REVISION~svn/
mv *$DAILY_REVISION*.deb /mnt/debian/dists/$DISTRIBUTION/main/binary-$ARCH/
mv *$DAILY_REVISION* /mnt/debian/dists/$DISTRIBUTION/main/source/

echo "Cleaning temporary build directory."
cd /mnt/debian/dists/
rm -rf "$BUILD_DIR"

echo "Removing old packages."
# (Versions of packages which have more then 5 versions and those old versions are more than one month old.)
perl -e "
for (<$DISTRIBUTION/main/binary-$ARCH/*.deb>) {
	m!.*/(.*?)_!;
	\$fs{\$1}++;
}
while ((\$f,\$n) = each(%fs)) {
	next if \$n <= 5;
	unlink for grep {-M > 30} <$DISTRIBUTION/main/binary-$ARCH/\$f_*.deb>;
	unlink for grep {-M > 30} <$DISTRIBUTION/main/source/\$f_*>;
}
"

echo "Making packages list."
dpkg-scanpackages --multiversion $DISTRIBUTION/main/binary-$ARCH /dev/null dists/ | gzip - > $DISTRIBUTION/main/binary-$ARCH/Packages.gz
dpkg-scansources $DISTRIBUTION/main/source /dev/null dists/ | gzip - > $DISTRIBUTION/main/source/Sources.gz

echo "Setting permissions."
chmod 644 $DISTRIBUTION/main/binary-$ARCH/* $DISTRIBUTION/main/source/*
