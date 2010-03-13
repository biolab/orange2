#!/bin/bash -e
#
# Should be run as: sudo ./debian-daily-build.sh [daily revision]
#

APT_ARGS="--assume-yes"
BUILD_DIR="/tmp/orange-build"

ARCH=`dpkg --print-architecture`

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

[ -e /mnt/debian/ ] || { echo "/mnt/debian/ not mounted."; exit 2; }

# Default is current latest revision in trunk
DAILY_REVISION=${2:-`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}
# svn info does not return proper exit status on an error so we check it this way
[ "$DAILY_REVISION" ] || exit 3

# Adds our repository to APT configuration
if ! grep -q "deb http://www.ailab.si/orange/debian lenny main" /etc/apt/sources.list; then
	echo "Adding ailab packages repository to APT configuration."
	echo "deb http://www.ailab.si/orange/debian lenny main" >> /etc/apt/sources.list
	echo "deb-src http://www.ailab.si/orange/debian lenny main" >> /etc/apt/sources.list
fi

if [ -e "/mnt/debian/dists/lenny/main/binary-$ARCH/python-orange-svn_0.0.$DAILY_REVISION-1_$ARCH.deb" ]; then
	echo "Package for $DAILY_REVISION revision already exists."
	exit 0
fi

apt-get $APT_ARGS update

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

apt-get $APT_ARGS source python-orange-svn

if [ -e "python-orange-svn-0.0.$DAILY_REVISION" ]; then
	echo "Package for $DAILY_REVISION revision already exists, just building it."
fi
else
	echo "Making source archive python-orange-svn-0.0.$DAILY_REVISION."
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/orange/ python-orange-svn-0.0.$DAILY_REVISION/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/source/ python-orange-svn-0.0.$DAILY_REVISION/source/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/add-ons/orngCRS/src/ python-orange-svn-0.0.$DAILY_REVISION/source/crs/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/COPYING python-orange-svn-0.0.$DAILY_REVISION/COPYING
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/LICENSES python-orange-svn-0.0.$DAILY_REVISION/LICENSES
	
	[ -e python-orange-svn-0.0.$DAILY_REVISION/doc/COPYING ] && mv python-orange-svn-0.0.$DAILY_REVISION/doc/COPYING python-orange-svn-0.0.$DAILY_REVISION/
	[ -e python-orange-svn-0.0.$DAILY_REVISION/doc/LICENSES ] && mv python-orange-svn-0.0.$DAILY_REVISION/doc/LICENSES python-orange-svn-0.0.$DAILY_REVISION/
	
	tar -czf python-orange-svn-0.0.$DAILY_REVISION.tar.gz python-orange-svn-0.0.$DAILY_REVISION
	rm -rf python-orange-svn-0.0.$DAILY_REVISION/
	
	echo "Updating packages."
	cd python-orange-svn-0.0.*/
	export DEBFULLNAME="Mitar"
	export DEBEMAIL="mitar@tnode.com"
	uupdate --no-symlink python-orange-svn-0.0.$DAILY_REVISION.tar.gz
	cd ..
	rm -rf python-orange-svn-0.0.$DAILY_REVISION.orig/ python-orange-svn-0.0.$DAILY_REVISION.tar.gz
	
	echo "Building new packages."
fi

cd python-orange-svn-0.0.$DAILY_REVISION/
dpkg-buildpackage -D -E -sa -us -uc

echo "Preparing public ailab Debian repository."
mkdir -p /mnt/debian/dists/lenny/main/binary-$ARCH/
mkdir -p /mnt/debian/dists/lenny/main/source/

echo "Copying to repository new packages."
cd ..
rm -rf python-orange-svn-0.0.$DAILY_REVISION/
mv *$DAILY_REVISION*.deb /mnt/debian/dists/lenny/main/binary-$ARCH/
mv *$DAILY_REVISION* /mnt/debian/dists/lenny/main/source/

echo "Cleaning temporary build directory."
cd /mnt/debian/dists/
rm -rf "$BUILD_DIR"

echo "Removing old packages."
# (Versions of packages which have more then 5 versions and those old versions are more than one month old.)
perl -e "
for (<lenny/main/binary-$ARCH/*.deb>) {
	m!.*/(.*?)_!;
	\$fs{\$1}++;
}
while ((\$f,\$n) = each(%fs)) {
	next if \$n <= 5;
	unlink for grep {-M > 30} <lenny/main/binary-$ARCH/\$f_*.deb>;
	unlink for grep {-M > 30} <lenny/main/source/\$f_*>;
}
"

echo "Making packages list."
dpkg-scanpackages --multiversion lenny/main/binary-$ARCH /dev/null dists/ | gzip - > lenny/main/binary-$ARCH/Packages.gz
dpkg-scansources lenny/main/source /dev/null dists/ | gzip - > lenny/main/source/Sources.gz

echo "Setting permissions."
chmod 644 lenny/main/binary-$ARCH/* lenny/main/source/*
