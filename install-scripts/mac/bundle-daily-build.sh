#!/bin/bash -e
#
# Should be run as: ./bundle-daily-build.sh [stable revision] [daily revision]
#

# Lists of add-ons to include
STABLE_ADDONS=""
DAILY_ADDONS="Bioinformatics Text"

# Sets error handler
trap "echo \"Script failed\"" ERR

[ -e /Volumes/download/ ] || { echo "/Volumes/download/ not mounted."; exit 1; }

if [ ! -x /usr/bin/xcodebuild ]; then
	echo "It seems Xcode is not installed on a system."
	exit 2
fi

# Defaults are current latest revisions in stable branch and trunk
STABLE_REVISION=${1:-`svn info --non-interactive http://www.ailab.si/svn/orange/branches/ver1.0/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}
DAILY_REVISION=${2:-`svn info --non-interactive http://www.ailab.si/svn/orange/trunk/ | grep 'Last Changed Rev:' | cut -d ' ' -f 4`}

echo "Preparing temporary directory."
rm -rf /private/tmp/bundle/

if [ ! -e /Volumes/download/orange-bundle-1.0b.$STABLE_REVISION.dmg ]; then
	echo "Downloading bundle template."
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/install-scripts/mac/bundle/ /private/tmp/bundle/
	
	echo "Downloading Orange stable source code revision $STABLE_REVISION."
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/orange/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/source/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/
	svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/add-ons/orngCRS/src/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/crs/
	
	[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/COPYING ] || svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/COPYING /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/COPYING
	[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/LICENSES ] || svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/LICENSES /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/LICENSES
	
	ln -s ../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/ /private/tmp/bundle/Orange.app/Contents/Resources/orange
	ln -s ../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/ /private/tmp/bundle/Orange.app/Contents/Resources/doc
	
	# Gives our Python executable to compile scripts later on
	export PATH=/private/tmp/bundle/Orange.app/Contents/MacOS/:$PATH
	
	echo "Compiling."
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/
	make
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/crs/
	make
	mv _orngCRS.so ../../
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/
	
	echo "Correcting install names for modules."
	for module in *.so ; do
		[ -L $module ] && continue
		
		install_name_tool -id @executable_path/../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/$module $module
		
		perl -MFile::Spec::Functions=abs2rel -e '
		for (`/usr/bin/otool -L -X $ARGV[0]`) {
			next unless m|^\s+(/private/tmp/bundle/Orange.app/.*) \(.*\)$|;
			system("/usr/bin/install_name_tool", "-change", $1, "\@loader_path/" . abs2rel($1), $ARGV[0]); 
		}
		' $module
	done
	
	echo "Cleaning up."
	rm -rf source/ c45.dll liborange_include.a
	
	# Installation registration
	echo "orange" > /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange.pth
	
	for addon in $STABLE_ADDONS ; do
		echo "Downloading Orange add-on $addon stable source code revision $STABLE_REVISION."
		svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/add-ons/$addon/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/
		
		[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/COPYING ] || svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/COPYING /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/COPYING
		[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/LICENSES ] || svn export --non-interactive --revision $STABLE_REVISION http://www.ailab.si/svn/orange/branches/ver1.0/LICENSES /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/LICENSES
		
		echo "Compiling add-on."
		cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/source/
		make
		cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/
		
		echo "Correcting install names for modules."
		for module in *.so ; do
			[ -L $module ] && continue
		
			install_name_tool -id @executable_path/../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/$module $module
			
			perl -MFile::Spec::Functions=abs2rel -e '
			for (`/usr/bin/otool -L -X $ARGV[0]`) {
				next unless m|^\s+(/private/tmp/bundle/Orange.app/.*) \(.*\)$|;
				system("/usr/bin/install_name_tool", "-change", $1, "\@loader_path/" . abs2rel($1), $ARGV[0]); 
			}
			' $module
		done
		
		echo "Cleaning up."
		rm -rf source/ setup.py
		
		# Installation registration
		echo "orange/add-ons/$addon" > /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange-`echo $addon | tr "[:upper:]" "[:lower:]"`.pth
	done
	
	echo "Removing unnecessary files."
	find /private/tmp/bundle/ \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' -or -name '*.pyd' \) -exec rm -rf {} ';'
	
	# Makes a link to Applications folder
	ln -s /Applications/ /private/tmp/bundle/Applications
	
	echo "Creating temporary image with the bundle."
	rm -f /private/tmp/bundle.dmg
	hdiutil detach /Volumes/Orange -force || true
	hdiutil create -format UDRW -volname Orange -fs HFS+ -fsargs "-c c=64,a=16,e=16" -srcfolder /private/tmp/bundle/ /private/tmp/bundle.dmg
	MOUNT_OUTPUT=`hdiutil attach -readwrite -noverify -noautoopen /private/tmp/bundle.dmg | egrep '^/dev/'`
	DEV_NAME=`echo -n "$MOUNT_OUTPUT" | head -n 1 | awk '{print $1}'`
	MOUNT_POINT=`echo -n "$MOUNT_OUTPUT" | tail -n 1 | awk '{print $3}'`
	
	# Makes the disk image window open automatically when mounted
	bless -openfolder "$MOUNT_POINT"
	# Hides background directory even more
	/Developer/Tools/SetFile -a V "$MOUNT_POINT/.background/"
	# Sets the custom icon volume flag so that volume has nice Orange icon after mount (.VolumeIcon.icns)
	/Developer/Tools/SetFile -a C "$MOUNT_POINT"
	
	rm -rf "$MOUNT_POINT/.Trashes/"
	rm -rf "$MOUNT_POINT/.fseventsd/"
	
	hdiutil detach "$DEV_NAME" -force
	
	echo "Converting temporary image to a compressed image."
	rm -f /private/tmp/orange-bundle-1.0b.$STABLE_REVISION.dmg
	hdiutil convert /private/tmp/bundle.dmg -format UDZO -imagekey zlib-level=9 -o /private/tmp/orange-bundle-1.0b.$STABLE_REVISION.dmg
	
	echo "Cleaning up."
	rm -f /private/tmp/bundle.dmg
	rm -rf /private/tmp/bundle/
fi

if [ ! -e /Volumes/download/orange-bundle-svn-0.0.$DAILY_REVISION.dmg ]; then
	echo "Downloading bundle template."
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/install-scripts/mac/bundle/ /private/tmp/bundle/
	
	echo "Downloading Orange daily source code revision $DAILY_REVISION."
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/orange/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/source/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/
	svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/add-ons/orngCRS/src/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/crs/
	
	[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/COPYING ] || svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/COPYING /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/COPYING
	[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/LICENSES ] || svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/LICENSES /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/LICENSES
	
	ln -s ../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/ /private/tmp/bundle/Orange.app/Contents/Resources/orange
	ln -s ../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/doc/ /private/tmp/bundle/Orange.app/Contents/Resources/doc
	
	echo "Compiling."
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/
	make
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/source/crs/
	make
	mv _orngCRS.so ../../
	cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/
	
	echo "Correcting install names for modules."
	for module in *.so ; do
		[ -L $module ] && continue
		
		install_name_tool -id @executable_path/../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/$module $module
		
		perl -MFile::Spec::Functions=abs2rel -e '
		for (`/usr/bin/otool -L -X $ARGV[0]`) {
			next unless m|^\s+(/private/tmp/bundle/Orange.app/.*) \(.*\)$|;
			system("/usr/bin/install_name_tool", "-change", $1, "\@loader_path/" . abs2rel($1), $ARGV[0]); 
		}
		' $module
	done
	
	echo "Cleaning up."
	rm -rf source/ c45.dll liborange_include.a
	
	# Installation registration
	echo "orange" > /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange.pth
	
	for addon in $DAILY_ADDONS ; do
		echo "Downloading Orange add-on $addon daily source code revision $DAILY_REVISION."
		svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/add-ons/$addon/ /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/
		
		[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/COPYING ] || svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/COPYING /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/COPYING
		[ -e /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/LICENSES ] || svn export --non-interactive --revision $DAILY_REVISION http://www.ailab.si/svn/orange/trunk/LICENSES /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/doc/LICENSES
		
		echo "Compiling add-on."
		cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/source/
		make
		cd /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/
		
		echo "Correcting install names for modules."
		for module in *.so ; do
			[ -L $module ] && continue
		
			install_name_tool -id @executable_path/../Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange/add-ons/$addon/$module $module
			
			perl -MFile::Spec::Functions=abs2rel -e '
			for (`/usr/bin/otool -L -X $ARGV[0]`) {
				next unless m|^\s+(/private/tmp/bundle/Orange.app/.*) \(.*\)$|;
				system("/usr/bin/install_name_tool", "-change", $1, "\@loader_path/" . abs2rel($1), $ARGV[0]); 
			}
			' $module
		done
		
		echo "Cleaning up."
		rm -rf source/ setup.py
		
		# Installation registration
		echo "orange/add-ons/$addon" > /private/tmp/bundle/Orange.app/Contents/Frameworks/Python.framework/Versions/2.5/lib/python2.5/site-packages/orange-`echo $addon | tr "[:upper:]" "[:lower:]"`.pth
	done
	
	echo "Removing unnecessary files."
	find /private/tmp/bundle/ \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' -or -name '*.pyd' \) -exec rm -rf {} ';'
	
	# Makes a link to Applications folder
	ln -s /Applications/ /private/tmp/bundle/Applications
	
	echo "Creating temporary image with the bundle."
	rm -f /private/tmp/bundle.dmg
	hdiutil detach /Volumes/Orange -force || true
	hdiutil create -format UDRW -volname Orange -fs HFS+ -fsargs "-c c=64,a=16,e=16" -srcfolder /private/tmp/bundle/ /private/tmp/bundle.dmg
	MOUNT_OUTPUT=`hdiutil attach -readwrite -noverify -noautoopen /private/tmp/bundle.dmg | egrep '^/dev/'`
	DEV_NAME=`echo -n "$MOUNT_OUTPUT" | head -n 1 | awk '{print $1}'`
	MOUNT_POINT=`echo -n "$MOUNT_OUTPUT" | tail -n 1 | awk '{print $3}'`
	
	# Makes the disk image window open automatically when mounted
	bless -openfolder "$MOUNT_POINT"
	# Hides background directory even more
	/Developer/Tools/SetFile -a V "$MOUNT_POINT/.background/"
	# Sets the custom icon volume flag so that volume has nice Orange icon after mount (.VolumeIcon.icns)
	/Developer/Tools/SetFile -a C "$MOUNT_POINT"
	
	rm -rf "$MOUNT_POINT/.Trashes/"
	rm -rf "$MOUNT_POINT/.fseventsd/"
	
	hdiutil detach "$DEV_NAME" -force
	
	echo "Converting temporary image to a compressed image."
	rm -f /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg
	hdiutil convert /private/tmp/bundle.dmg -format UDZO -imagekey zlib-level=9 -o /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg
	
	echo "Cleaning up."
	rm -f /private/tmp/bundle.dmg
	rm -rf /private/tmp/bundle/
fi

echo "Removing old versions of bundles."
# (Keeps last 5 versions.)
perl -e 'unlink ((reverse sort </Volumes/download/orange-bundle-svn-0*.dmg>)[5..10000])'
perl -e 'unlink ((reverse sort </Volumes/download/orange-bundle-1*.dmg>)[5..10000])'

if [ -e /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg ] || [ -e /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg ]; then
	echo "Moving bundles to the download directory."
	[ -e /private/tmp/orange-bundle-1.0b.$STABLE_REVISION.dmg ] && mv /private/tmp/orange-bundle-1.0b.$STABLE_REVISION.dmg /Volumes/download/
	[ -e /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg ] && mv /private/tmp/orange-bundle-svn-0.0.$DAILY_REVISION.dmg /Volumes/download/
	
	echo "Registering new bundles."
	egrep -v '^(MAC_STABLE|MAC_DAILY)=' /Volumes/download/filenames.set > /Volumes/download/filenames.set.new
	echo "MAC_STABLE=orange-bundle-1.0b.$STABLE_REVISION.dmg" >> /Volumes/download/filenames.set.new
	echo "MAC_DAILY=orange-bundle-svn-0.0.$DAILY_REVISION.dmg" >> /Volumes/download/filenames.set.new
	mv /Volumes/download/filenames.set.new /Volumes/download/filenames.set
fi
