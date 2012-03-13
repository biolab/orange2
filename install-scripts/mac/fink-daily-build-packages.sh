#!/bin/bash -e
#
# Run daily fink build
# 

# Daily orange packages to build
DAILY_PACKAGES="orange-hg-py26 orange-gui-hg-py26 orange-bioinformatics-hg-py26 orange-bioinformatics-gui-hg-py26 orange-text-hg-py26 orange-text-gui-hg-py26 orange-hg-py27 orange-gui-hg-py27 orange-bioinformatics-hg-py27 orange-bioinformatics-gui-hg-py27 orange-text-hg-py27 orange-text-gui-hg-py27"

# Packages which, when installing, want special confirmation from the user
# We keep those packages installed all the time
SPECIAL_PACKAGES="passwd xinitrc"

# A list of packages (dependencies) from which user can choose upon installing our packages
# We would like to build all those so that it does not need to compile anything whichever packages he or she chooses
# The problem is that they are often mutually conflicting so we cannot have them simply installed (so that update-all
# would update them) but have to build them explicitly
OTHER_PACKAGES="ghostscript ghostscript-esp ghostscript6 ghostscript-nox ghostscript6-nox gnuplot gnuplot-nox gnuplot-nogtk tetex-base tetex-nox-base texlive-nox-base texlive-base tetex-texmf texlive-texmf"


# Miscellaneous extra packages which are maybe not really needed for Orange but are useful for CS research
EXTRA_PACKAGES="fuse gcc42 gcc43 gcc44 gnuplot gnuplot-nox gnuplot-nogtk db48 db48-aes git imagemagick-nox rrdtool maxima nmap wireshark openssl pstree python26 python27 python3 rdiff-backup svn swi-prolog lynx links w3m elinks matplotlib-py26 matplotlib-py27 mercurial-py26 mercurial-py27"

FINK_ARGS="--yes --build-as-nobody"
FINK_SELFUPDATE_ARGS="--yes"
APT_ARGS="--assume-yes"

# Path to Fink root
FINK_ROOT=/sw

# Repo dir
REPO_DIR=/private/tmp/repos

ARCH=`perl -MFink::FinkVersion -e 'print Fink::FinkVersion::get_arch'`

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

test -r $FINK_ROOT/bin/init.sh || { echo "Fink cannot be found." exit 2; }

[ -e /Volumes/fink/ ] || { echo "/Volumes/fink/ not mounted."; exit 3; }

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

if [ ! "`/usr/X11/bin/X -version 2>&1 | grep '^X.Org X Server' | grep -E -o '[0-9]+\.[0-9]+\.[0-9]+' | cut -d '.' -f 2`" -gt "3" ]; then
	echo "It seems X11 version 2.3.0 or later is not installed on a system."
	exit 8
fi

echo "Preparing local biolab Fink info files repository."
mkdir -p $FINK_ROOT/fink/dists/biolab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/*

echo "Updating local biolab Fink info files repository."
curl "http://orange.biolab.si/fink/dists/10.$MAC_VERSION/main/finkinfo/all.tgz" --output $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz
tar -xzf $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz -C $FINK_ROOT/fink/dists/biolab/main/finkinfo/
rm -f $FINK_ROOT/fink/dists/biolab/main/finkinfo/all.tgz

# Copy info files from local/main/finkinfo
echo "Updating new fink info files."
mv $FINK_ROOT/fink/dists/local/main/finkinfo/*.info $FINK_ROOT/fink/dists/biolab/main/finkinfo/

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
fink $FINK_ARGS purge --recursive $DAILY_PACKAGES $OTHER_PACKAGES $EXTRA_PACKAGES
# Sometimes Fink and APT are not in sync so we remove packages also directly
for package in $DAILY_PACKAGES $OTHER_PACKAGES $EXTRA_PACKAGES ; do
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
for package in $DAILY_PACKAGES ; do
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
	next if \$versions{\$1} or -M() < 30;
	unlink;
}
"

