#!/bin/bash -e
#
# Should be run as: sudo ./fink-daily-build.sh [stable revision] [daily revision]
#
# If [stable revision] and/or [daily revision] is specified it makes source archives and updates
# our info files with those revisions (if any is not specified it uses its latest revision)
# before building Fink packages
#

# Those packages should not be installed as we are just building them (and dependencies)
# The order is important as for validation to work we should first build packages which do not depend on other
# packages we want to validate (as they would be build without validation as dependencies)
STABLE_PACKAGES="orange-py25 orange"
DAILY_PACKAGES="orange-svn-py26 orange-gui-svn-py26 orange-bioinformatics-svn-py26 orange-bioinformatics-gui-svn-py26 orange-text-svn-py26 orange-text-gui-svn-py26 orange-svn-py27 orange-gui-svn-py27 orange-bioinformatics-svn-py27 orange-bioinformatics-gui-svn-py27 orange-text-svn-py27 orange-text-gui-svn-py27"

LAST_MAC_VERSION_FOR_STABLE_PACKAGES=5

# Packages which, when installing, want special confirmation from the user
# We keep those packages installed all the time
SPECIAL_PACKAGES="passwd xinitrc"

# Additional source directories which get packed
STABLE_SOURCE_DIRS="install-scripts/mac/bundle-lite/"
DAILY_SOURCE_DIRS="install-scripts/mac/bundle-lite/ add-ons/Bioinformatics/ add-ons/Text/"

# A list of packages (dependencies) from which user can choose upon installing our packages
# We would like to build all those so that it does not need to compile anything whichever packages he or she chooses
# The problem is that they are often mutually conflicting so we cannot have them simply installed (so that update-all
# would update them) but have to build them explicitly
OTHER_PACKAGES="ghostscript ghostscript-esp ghostscript6 ghostscript-nox ghostscript6-nox fltk-backend-aqua-oct324 fltk-backend-x11-oct324 gnuplot gnuplot-nox gnuplot-nogtk tetex-base tetex-nox-base texlive-nox-base texlive-base tetex-texmf"

# Miscellaneous extra packages which are maybe not really needed for Orange but are useful for CS research
EXTRA_PACKAGES="fuse gcc42 gcc43 gcc44 gnuplot gnuplot-nox gnuplot-nogtk octave db48 db48-aes git imagemagick-nox rrdtool maxima nmap wireshark openssl pstree python26 python27 python3 rdiff-backup svn swi-prolog lynx links w3m elinks matplotlib-py25 matplotlib-py26 matplotlib-py27"

FINK_ARGS="--yes --build-as-nobody"
FINK_SELFUPDATE_ARGS="--yes"
APT_ARGS="--assume-yes"

# Path to Fink root
FINK_ROOT=/sw

ARCH=`perl -MFink::FinkVersion -e 'print Fink::FinkVersion::get_arch'`

if [ "$1" ] || [ "$2" ]; then
	PACKAGE_SOURCE=1
fi

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

test -r $FINK_ROOT/bin/init.sh || { echo "Fink cannot be found." exit 2; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 3; }

if [ $PACKAGE_SOURCE ]; then
	[ -e /Volumes/download/ ] || { echo "/Volumes/download/ not mounted."; exit 4; }
fi

# Configures environment for Fink
. $FINK_ROOT/bin/init.sh

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'unstable/main' && grep '^SelfUpdateMethod:' $FINK_ROOT/etc/fink.conf | grep -q 'point'; then
	echo "Fink does not seem to use unstable Fink packages tree with rsync or CVS updating."
	exit 5
fi

if [ ! -x /usr/bin/xcodebuild ]; then
	echo "It seems Xcode is not installed on a system."
	exit 6
fi

MAC_VERSION=`sw_vers -productVersion | cut -d '.' -f 2`
if [[ "$MAC_VERSION" -ne 5 && "$MAC_VERSION" -ne 6 ]]; then
	echo "It seems system is not Mac OS X version 10.5 or 10.6."
	exit 7
fi
if [[ "$MAC_VERSION" -gt "$LAST_MAC_VERSION_FOR_STABLE_PACKAGES" ]]; then
	STABLE_PACKAGES=""
fi

if [ ! "`/usr/X11/bin/X -version 2>&1 | grep '^X.Org X Server' | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+' | cut -d '.' -f 2`" -gt "3" ]; then
	echo "It seems X11 version 2.3.0 or later is not installed on a system."
	exit 8
fi

if [ $PACKAGE_SOURCE ]; then
	# Defaults are current latest revisions in stable branch and trunk
	STABLE_REVISION=${1:-`svn info --non-interactive http://orange.biolab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}
	# svn info does not return proper exit status on an error so we check it this way
	[ "$STABLE_REVISION" ] || exit 9
	DAILY_REVISION=${2:-`svn info --non-interactive http://orange.biolab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}
	# svn info does not return proper exit status on an error so we check it this way
	[ "$DAILY_REVISION" ] || exit 10
fi

echo "Preparing local biolab Fink info files repository."
mkdir -p $FINK_ROOT/fink/dists/biolab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/*
if [ $PACKAGE_SOURCE ]; then	
	# Gets Fink package info files from SVN
	svn export --force --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/install-scripts/mac/fink/ $FINK_ROOT/fink/dists/biolab/main/finkinfo/
	
	# Injects revision versions into templates
	perl -pi -e "s/__STABLE_REVISION__/$STABLE_REVISION/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
	perl -pi -e "s/__DAILY_REVISION__/$DAILY_REVISION/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
else
	# Gets current (daily) info files from SVN
	echo "Updating local biolab Fink info files repository."
	curl "http://orange.biolab.si/fink/dists/10.$MAC_VERSION/main/finkinfo/all.tgz" --output $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz
	tar -xzf $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz -C $FINK_ROOT/fink/dists/biolab/main/finkinfo/
	rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz
fi

if ! grep '^Trees:' $FINK_ROOT/etc/fink.conf | grep -q 'biolab/main'; then
	echo "Adding local biolab Fink info files repository to Fink configuration."
	perl -p -i -l -e '$_ = "$_ biolab/main" if /^Trees/' $FINK_ROOT/etc/fink.conf
fi

# Adds our binary repository to local Fink (APT) configuration
if ! grep -q "deb http://orange.biolab.si/fink 10.$MAC_VERSION main" $FINK_ROOT/etc/apt/sources.list; then
	echo "Adding biolab Fink binary packages repository to Fink configuration."
	echo "deb http://orange.biolab.si/fink 10.$MAC_VERSION main" >> $FINK_ROOT/etc/apt/sources.list
fi

if [ ! -e $FINK_ROOT/etc/apt/apt.conf.d/daily-build ]; then
	echo "Configuring apt-get to assume yes to all questions."
	echo 'APT::Get::Assume-Yes "true";' > $FINK_ROOT/etc/apt/apt.conf.d/daily-build
fi

if [ $PACKAGE_SOURCE ]; then
	mkdir -p "/Volumes/fink/dists/10.$MAC_VERSION/main/source/"
	chmod +rx "/Volumes/fink/dists/10.$MAC_VERSION/main/source/"
	
	if [ ! -e /Volumes/fink/dists/10.$MAC_VERSION/main/source/orange-1.0b.$STABLE_REVISION.tgz ]; then
		echo "Making source archive orange-1.0b.$STABLE_REVISION."
		
		rm -rf /tmp/orange-1.0b.$STABLE_REVISION/ /tmp/orange-1.0b.$STABLE_REVISION.tgz
		
		svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/orange/ /tmp/orange-1.0b.$STABLE_REVISION/
		svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/source/ /tmp/orange-1.0b.$STABLE_REVISION/source/
		svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/add-ons/orngCRS/src/ /tmp/orange-1.0b.$STABLE_REVISION/source/crs/
		svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/COPYING /tmp/orange-1.0b.$STABLE_REVISION/COPYING
		svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/LICENSES /tmp/orange-1.0b.$STABLE_REVISION/LICENSES
		
		[ -e /tmp/orange-1.0b.$STABLE_REVISION/doc/COPYING ] && mv /tmp/orange-1.0b.$STABLE_REVISION/doc/COPYING /tmp/orange-1.0b.$STABLE_REVISION/
		[ -e /tmp/orange-1.0b.$STABLE_REVISION/doc/LICENSES ] && mv /tmp/orange-1.0b.$STABLE_REVISION/doc/LICENSES /tmp/orange-1.0b.$STABLE_REVISION/
		
		tar -czf /tmp/orange-1.0b.$STABLE_REVISION.tgz -C /tmp/ orange-1.0b.$STABLE_REVISION
		
		MD5SUM=`md5 -q /tmp/orange-1.0b.$STABLE_REVISION.tgz`
		
		mv /tmp/orange-1.0b.$STABLE_REVISION.tgz /Volumes/fink/dists/10.$MAC_VERSION/main/source/
		chmod -R +r "/Volumes/fink/dists/10.$MAC_VERSION/main/source/"
		
		rm -rf /tmp/orange-1.0b.$STABLE_REVISION/
		
		echo "Registering new source archive."
		egrep -v '^SOURCE_STABLE=' /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
		echo "SOURCE_STABLE=orange-1.0b.$STABLE_REVISION.tgz" >> /Volumes/download/filenames_mac.set.new
		mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set
		chmod +r /Volumes/download/filenames_mac.set
	else
		MD5SUM=`md5 -q /Volumes/fink/dists/10.$MAC_VERSION/main/source/orange-1.0b.$STABLE_REVISION.tgz`
	fi
	
	perl -pi -e "s/__STABLE_MD5SUM_ORANGE__/$MD5SUM/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
	
	if [ ! -e /Volumes/fink/dists/10.$MAC_VERSION/main/source/orange-svn-0.0.$DAILY_REVISION.tgz ]; then
		echo "Making source archive orange-svn-0.0.$DAILY_REVISION."
		
		rm -rf /tmp/orange-svn-0.0.$DAILY_REVISION/ /tmp/orange-svn-0.0.$DAILY_REVISION.tgz
		
		svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/orange/ /tmp/orange-svn-0.0.$DAILY_REVISION/
		svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/source/ /tmp/orange-svn-0.0.$DAILY_REVISION/source/
		svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/add-ons/orngCRS/src/ /tmp/orange-svn-0.0.$DAILY_REVISION/source/crs/
		svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/COPYING /tmp/orange-svn-0.0.$DAILY_REVISION/COPYING
		svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/LICENSES /tmp/orange-svn-0.0.$DAILY_REVISION/LICENSES
		
		[ -e /tmp/orange-svn-0.0.$DAILY_REVISION/doc/COPYING ] && mv /tmp/orange-svn-0.0.$DAILY_REVISION/doc/COPYING /tmp/orange-svn-0.0.$DAILY_REVISION/
		[ -e /tmp/orange-svn-0.0.$DAILY_REVISION/doc/LICENSES ] && mv /tmp/orange-svn-0.0.$DAILY_REVISION/doc/LICENSES /tmp/orange-svn-0.0.$DAILY_REVISION/
		
		tar -czf /tmp/orange-svn-0.0.$DAILY_REVISION.tgz -C /tmp/ orange-svn-0.0.$DAILY_REVISION
		
		MD5SUM=`md5 -q /tmp/orange-svn-0.0.$DAILY_REVISION.tgz`
		
		mv /tmp/orange-svn-0.0.$DAILY_REVISION.tgz /Volumes/fink/dists/10.$MAC_VERSION/main/source/
		chmod -R +r /Volumes/fink/dists/10.$MAC_VERSION/main/source/
		
		rm -rf /tmp/orange-svn-0.0.$DAILY_REVISION/
		
		echo "Registering new source archive."
		egrep -v '^SOURCE_DAILY=' /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
		echo "SOURCE_DAILY=orange-svn-0.0.$DAILY_REVISION.tgz" >> /Volumes/download/filenames_mac.set.new
		mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set
		chmod +r /Volumes/download/filenames_mac.set
	else
		MD5SUM=`md5 -q /Volumes/fink/dists/10.$MAC_VERSION/main/source/orange-svn-0.0.$DAILY_REVISION.tgz`
	fi
	
	perl -pi -e "s/__DAILY_MD5SUM_ORANGE__/$MD5SUM/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
	
	for dir in $STABLE_SOURCE_DIRS ; do
		# Gets only the last part of the directory name, converts to lower case and removes dashes
		SOURCE_NAME=`basename $dir | tr "[:upper:]" "[:lower:]" | tr -d "-"`
		SOURCE_VAR=`basename $dir | tr "[:lower:]" "[:upper:]" | tr -d "-"`
		STABLE_SOURCE_NAME=orange-$SOURCE_NAME-1.0b.$STABLE_REVISION
		
		if [ ! -e /Volumes/fink/dists/10.$MAC_VERSION/main/source/$STABLE_SOURCE_NAME.tgz ]; then
			echo "Making source archive $STABLE_SOURCE_NAME."
			
			rm -rf /tmp/$STABLE_SOURCE_NAME/ /tmp/$STABLE_SOURCE_NAME.tgz
			
			svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/$dir /tmp/$STABLE_SOURCE_NAME/
			svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/COPYING /tmp/$STABLE_SOURCE_NAME/COPYING
			svn export --non-interactive --revision $STABLE_REVISION http://orange.biolab.si/svn/orange/branches/ver1.0/LICENSES /tmp/$STABLE_SOURCE_NAME/LICENSES
			
			[ -e /tmp/$STABLE_SOURCE_NAME/doc/COPYING ] && mv /tmp/$STABLE_SOURCE_NAME/doc/COPYING /tmp/$STABLE_SOURCE_NAME/
			[ -e /tmp/$STABLE_SOURCE_NAME/doc/LICENSES ] && mv /tmp/$STABLE_SOURCE_NAME/doc/LICENSES /tmp/$STABLE_SOURCE_NAME/
			
			tar -czf /tmp/$STABLE_SOURCE_NAME.tgz -C /tmp/ $STABLE_SOURCE_NAME
			
			MD5SUM=`md5 -q /tmp/$STABLE_SOURCE_NAME.tgz`
			
			mv /tmp/$STABLE_SOURCE_NAME.tgz /Volumes/fink/dists/10.$MAC_VERSION/main/source/
		
			rm -rf /tmp/$STABLE_SOURCE_NAME/
			
			echo "Registering new source archive."
			egrep -v "^SOURCE_${SOURCE_VAR}_STABLE=" /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
			echo "SOURCE_${SOURCE_VAR}_STABLE=$STABLE_SOURCE_NAME.tgz" >> /Volumes/download/filenames_mac.set.new
			mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set
		else
			MD5SUM=`md5 -q /Volumes/fink/dists/10.$MAC_VERSION/main/source/$STABLE_SOURCE_NAME.tgz`
		fi
		
		perl -pi -e "s/__STABLE_MD5SUM_\U$SOURCE_NAME\E__/$MD5SUM/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
	done
	
	for dir in $DAILY_SOURCE_DIRS ; do
		# Gets only the last part of the directory name, converts to lower case and removes dashes
		SOURCE_NAME=`basename $dir | tr "[:upper:]" "[:lower:]" | tr -d "-"`
		SOURCE_VAR=`basename $dir | tr "[:lower:]" "[:upper:]" | tr -d "-"`
		DAILY_SOURCE_NAME=orange-$SOURCE_NAME-svn-0.0.$DAILY_REVISION
		
		if [ ! -e /Volumes/fink/dists/10.$MAC_VERSION/main/source/$DAILY_SOURCE_NAME.tgz ]; then
			echo "Making source archive $DAILY_SOURCE_NAME."
			
			rm -rf /tmp/$DAILY_SOURCE_NAME/ /tmp/$DAILY_SOURCE_NAME.tgz
			
			svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/$dir /tmp/$DAILY_SOURCE_NAME/
			svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/COPYING /tmp/$DAILY_SOURCE_NAME/COPYING
			svn export --non-interactive --revision $DAILY_REVISION http://orange.biolab.si/svn/orange/trunk/LICENSES /tmp/$DAILY_SOURCE_NAME/LICENSES
			
			[ -e /tmp/$DAILY_SOURCE_NAME/doc/COPYING ] && mv /tmp/$DAILY_SOURCE_NAME/doc/COPYING /tmp/$DAILY_SOURCE_NAME/
			[ -e /tmp/$DAILY_SOURCE_NAME/doc/LICENSES ] && mv /tmp/$DAILY_SOURCE_NAME/doc/LICENSES /tmp/$DAILY_SOURCE_NAME/
			
			tar -czf /tmp/$DAILY_SOURCE_NAME.tgz -C /tmp/ $DAILY_SOURCE_NAME
			
			MD5SUM=`md5 -q /tmp/$DAILY_SOURCE_NAME.tgz`
			
			mv /tmp/$DAILY_SOURCE_NAME.tgz /Volumes/fink/dists/10.$MAC_VERSION/main/source/
		
			rm -rf /tmp/$DAILY_SOURCE_NAME/
			
			echo "Registering new source archive."
			egrep -v "^SOURCE_${SOURCE_VAR}_DAILY=" /Volumes/download/filenames_mac.set > /Volumes/download/filenames_mac.set.new
			echo "SOURCE_${SOURCE_VAR}_DAILY=$DAILY_SOURCE_NAME.tgz" >> /Volumes/download/filenames_mac.set.new
			mv /Volumes/download/filenames_mac.set.new /Volumes/download/filenames_mac.set
		else
			MD5SUM=`md5 -q /Volumes/fink/dists/10.$MAC_VERSION/main/source/$DAILY_SOURCE_NAME.tgz`
		fi
		
		perl -pi -e "s/__DAILY_MD5SUM_\U$SOURCE_NAME\E__/$MD5SUM/g" $FINK_ROOT/fink/dists/biolab/main/finkinfo/*.info
	done
fi

# Configures any pending packages from possible interrupted past sessions
dpkg --configure -a

# Gets all official Fink package info files
echo "Updating installed Fink packages."
fink $FINK_SELFUPDATE_ARGS selfupdate --method=rsync
fink $FINK_ARGS scanpackages

# Updates everything (probably by compiling new packages)
fink $FINK_ARGS update-all

# Installs special packages (if they are not already installed)
yes | fink $FINK_ARGS install $SPECIAL_PACKAGES

# Removes possiblly installed packages which we want built
fink $FINK_ARGS purge --recursive $STABLE_PACKAGES $DAILY_PACKAGES $OTHER_PACKAGES $EXTRA_PACKAGES
# Sometimes Fink and APT are not in sync so we remove packages also directly
for package in $STABLE_PACKAGES $DAILY_PACKAGES $OTHER_PACKAGES $EXTRA_PACKAGES ; do
	echo $package "purge" | dpkg --set-selections
done
apt-get $APT_ARGS dselect-upgrade

# Stores current packages status
dpkg --get-selections '*' > /tmp/dpkg-selections.list

for package in $OTHER_PACKAGES ; do
	# Restores intitial packages status
	dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
	dpkg --set-selections < /tmp/dpkg-selections.list
	apt-get $APT_ARGS dselect-upgrade
	
	# Builds a package if it has not been rebuilt already (for example, as a dependency)
	# We install it and not just build it because installation does not build package if it already exists as a binary package
	echo "Specially building package $package."
	fink $FINK_ARGS install $package
done

for package in $EXTRA_PACKAGES ; do
	if fink $FINK_ARGS describe $package > /dev/null ; then
		# Restores intitial packages status
		dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
		dpkg --set-selections < /tmp/dpkg-selections.list
		apt-get $APT_ARGS dselect-upgrade
		
		# Builds a package if it has not been rebuilt already (for example, as a dependency)
		# We install it and not just build it because installation does not build package if it already exists as a binary package
		echo "Specially building extra package $package."
		fink $FINK_ARGS install $package
	else
		echo "Not building extra package $package."
	fi
done

# We build our packages in "maintainer" mode - Fink makes tests and validates packages
for package in $STABLE_PACKAGES $DAILY_PACKAGES ; do
	DEPS=`perl -MFink -MFink::PkgVersion -l -e "Fink::Package->require_packages(); map { map { /(\\S+)/; print \\$1 } @\\$_ } @{Fink::PkgVersion->match_package('$package')->get_depends(1, 0)};"`
	
	# First builds all dependencies normally (so that we are not checking for others' errors)
	for deps in $DEPS ; do
		# Restores intitial packages status
		dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
		dpkg --set-selections < /tmp/dpkg-selections.list
		apt-get $APT_ARGS dselect-upgrade
		
		# We install it and not just build it because installation does not build package if it already exists as a binary package
		echo "Specially building package $package dependency $deps."
		fink $FINK_ARGS install $deps
	done
	
	# Restores intitial packages status
	dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
	dpkg --set-selections < /tmp/dpkg-selections.list
	apt-get $APT_ARGS dselect-upgrade
	
	# Then builds a package
	# We can just build it as our packages have been probably cached if they have been already built
	echo "Specially building, testing and validating package $package."
	fink $FINK_ARGS --maintainer build $package
done

echo "Restoring initial packages status."
dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
dpkg --set-selections < /tmp/dpkg-selections.list
apt-get $APT_ARGS dselect-upgrade
rm -f /tmp/dpkg-selections.list

# Cleans unncessary files (we cache them anyway in public repository)
echo "Cleaning."
fink $FINK_ARGS cleanup --all

# TODO: Should be called only on a daily build server and not if building locally
/Users/ailabc/mount-dirs.sh

echo "Preparing public biolab Fink info and binary files repository."
mkdir -p /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/
chmod +rx /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/
mkdir -p /Volumes/fink/dists/10.$MAC_VERSION/main/finkinfo/
chmod +rx /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/

echo "Copying to repository all binary packages."
cp $FINK_ROOT/fink/debs/*.deb /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/
if (shopt -s nullglob; f=($FINK_ROOT/var/cache/apt/archives/*.deb); ((${#f[@]}))); then
	# We have to test if there are any deb files available as otherwise cp fails
	cp $FINK_ROOT/var/cache/apt/archives/*.deb /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/
fi

cd /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/

echo "Fixing possible problems with binary packages filenames."
# Some packages include Fink epoch which uses colon as a delimiter and breaks package retrieval from the repository web server
# We remove epoch as it should not be there in the first place
perl -e '
for (<*.deb>) {
	if (m/^(.+)_\d+%3a(.+)$/) {
		rename $_, "$1_$2";
	}
}
'

echo "Removing old binary packages."
# (Versions of packages which have more then 5 versions and those old versions are more than one month old.)
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
perl -MFink::Scanpackages -e "Fink::Scanpackages->scan('dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/');" | gzip - > dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/Packages.gz

echo "Archive: biolab
Origin: Fink
Component: main
Architecture: darwin-$ARCH
Label: Fink" > dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/Release

echo "Setting permissions."
chmod -R +r /Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/

echo "Copying to repository all info files."
rm -f /Volumes/fink/dists/10.$MAC_VERSION/main/finkinfo/*
cp $FINK_ROOT/fink/dists/biolab/main/finkinfo/* /Volumes/fink/dists/10.$MAC_VERSION/main/finkinfo/

echo "Making an archive of all info files."
cd $FINK_ROOT/fink/dists/biolab/main/finkinfo/
tar -czf /Volumes/fink/dists/10.$MAC_VERSION/main/finkinfo/all.tgz *

echo "Setting permissions."
chmod -R +r /Volumes/fink/dists/10.$MAC_VERSION/main/finkinfo/

echo "Removing unnecessary source archives."
perl -e "
for (</Volumes/fink/dists/10.$MAC_VERSION/main/binary-darwin-$ARCH/orange-*.deb>) {
	m/_(.+)-\\d+_darwin-$ARCH\\.deb/;
	\$versions{\$1} = 1;
}
for (</Volumes/fink/dists/10.$MAC_VERSION/main/source/*.tgz>) {
	m/.+-(.+)\\.tgz/;
	next if \$versions{\$1};
	unlink;
}
"
