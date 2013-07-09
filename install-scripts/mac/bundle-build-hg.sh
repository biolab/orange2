#!/bin/bash -e
#
# Build the orange Mac OSX bundle
#
# ./bundle-build-hg.sh work_dir revision bundle_output_file
# ./bundle-build-hg.sh /private/tmp tip /private/tmp/orange-bundle-hg-tip.dmg
#

WORK_DIR=$1
REVISION=$2
BUNDLE=$3

TMP_BUNDLE_DIR=${WORK_DIR}/bundle
REPOS_DIR=${WORK_DIR}/repos

# Remove leftovers if any
if [ -e $TMP_BUNDLE_DIR ]; then
	rm -rf $TMP_BUNDLE_DIR
fi

echo "Preaparing the bundle template"
TEMPLATE_VERSION=$(curl --silent http://orange.biolab.si/download/bundle-templates/CURRENT.txt)
curl --silent http://orange.biolab.si/download/bundle-templates/Orange-template-${TEMPLATE_VERSION}.tar.gz | tar -xz -C $WORK_DIR

# Make repos dir if it does not yet exist
if [ ! -e $REPOS_DIR ]; then
	mkdir $REPOS_DIR
fi

APP=${TMP_BUNDLE_DIR}/Orange.app

# Python interpreter in the bundle
PYTHON=${APP}/Contents/MacOS/python

# Python version
PY_VER=`$PYTHON -c "import sys; print sys.version[:3]"`

SITE_PACKAGES=${APP}/Contents/Frameworks/Python.framework/Versions/${PY_VER}/lib/python${PY_VER}/site-packages/

# easy_install script in the bundle
EASY_INSTALL=${APP}/Contents/MacOS/easy_install

# Link Python.app startup script to top bundle
ln -fs ../Frameworks/Python.framework/Versions/Current/Resources/Python.app/Contents/MacOS/Python ${APP}/Contents/MacOS/PythonAppStart

echo "Preparing startup scripts"

# Create an enironment startup script
cat <<-'EOF' > $APP/Contents/MacOS/ENV
	# Create an environment for running python from the bundle
	# Should be run as "source ENV"

	BUNDLE_DIR=`dirname "$0"`/../
	BUNDLE_DIR=`perl -MCwd=realpath -e 'print realpath($ARGV[0])' "$BUNDLE_DIR"`/
	FRAMEWORKS_DIR="$BUNDLE_DIR"Frameworks/
	RESOURCES_DIR="$BUNDLE_DIR"Resources/

	PYVERSION="2.7"

	PYTHONEXECUTABLE="$FRAMEWORKS_DIR"Python.framework/Resources/Python.app/Contents/MacOS/Python
	PYTHONHOME="$FRAMEWORKS_DIR"Python.framework/Versions/"$PYVERSION"/

	DYLD_FRAMEWORK_PATH="$FRAMEWORKS_DIR"${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}

	export PYTHONEXECUTABLE
	export PYTHONHOME

	export DYLD_FRAMEWORK_PATH

	# Some non framework libraries are put in $FRAMEWORKS_DIR by macho_standalone
	export DYLD_LIBRARY_PATH="$FRAMEWORKS_DIR"${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
EOF

# Create Orange application startup script
cat <<-'EOF' > ${APP}/Contents/MacOS/Orange
	#!/bin/bash

	DIRNAME=$(dirname "$0")
	source "$DIRNAME"/ENV

	# LaunchServices passes the Carbon process identifier to the application with
	# -psn parameter - we do not want it
	if [[ $1 == -psn_* ]]; then
	    shift 1
	fi

	exec -a "$0" "$DIRNAME"/PythonAppStart -m Orange.OrangeCanvas.main "$@"
EOF

chmod +x ${APP}/Contents/MacOS/Orange


echo "Checkouting and building orange"
echo "==============================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange orange $REVISION $REPOS_DIR ${APP}

echo "Specifically building orangeqt"
echo "------------------------------"

pushd $REPOS_DIR/orange/source/orangeqt
echo "Fixing sip/pyqt configuration"

sed -i.bak "s@/Users/.*/Orange.app/@$APP/@g" ${SITE_PACKAGES}/PyQt4/pyqtconfig.py
sed -i.bak "s@/Users/.*/Orange.app/@$APP/@g" ${SITE_PACKAGES}/sipconfig.py
export PATH=$APP/Contents/Resources/Qt4/bin:$PATH
$PYTHON setup.py install

popd

echo "Fixing Qt plugins search path"
echo "[Paths]
Plugins = ../../../../../Resources/Qt4/plugins/" > $APP/Contents/Frameworks/Python.framework/Resources/Python.app/Contents/Resources/qt.conf

echo "[Paths]
Plugins = Resources/Qt4/plugins/" > $APP/Contents/Resources/qt.conf


echo "Checkouting and building bioinformatics addon"
echo "============================================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange-bioinformatics bioinformatics tip $REPOS_DIR ${APP}

echo "Checkouting and building text addon"
echo "==================================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange-text text tip $REPOS_DIR ${APP}

echo "Instaling slumber library"
echo "+++++++++++++++++++++++++"
$EASY_INSTALL slumber

echo "Removing unnecessary files."
find $TMP_BUNDLE_DIR \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' -or -name '*.pyd' \) -exec rm -rf {} ';'

ln -s ../Frameworks/Python.framework/Versions/Current/lib/python${PY_VER}/site-packages/Orange ${APP}/Contents/Resources/Orange

	
echo "Preparing the .dmg image"
echo "========================"

# Makes a link to Applications folder
ln -s /Applications/ $TMP_BUNDLE_DIR/Applications

echo "Fixing bundle permissions."

{ chown -Rh root:wheel $TMP_BUNDLE_DIR; } || { echo "Could not fix bundle permissions"; }

echo "Creating temporary image with the bundle."

TMP_BUNDLE=${WORK_DIR}/bundle.dmg
rm -f $TMP_BUNDLE

hdiutil detach /Volumes/Orange -force || true
hdiutil create -format UDRW -volname Orange -fs HFS+ -fsargs "-c c=64,a=16,e=16" -srcfolder $TMP_BUNDLE_DIR $TMP_BUNDLE
MOUNT_OUTPUT=`hdiutil attach -readwrite -noverify -noautoopen $TMP_BUNDLE | egrep '^/dev/'`
DEV_NAME=`echo -n "$MOUNT_OUTPUT" | head -n 1 | awk '{print $1}'`
MOUNT_POINT=`echo -n "$MOUNT_OUTPUT" | tail -n 1 | awk '{print $3}'`

# Makes the disk image window open automatically when mounted
bless -openfolder "$MOUNT_POINT"
# Hides background directory even more
/Developer/Tools/SetFile -a V "$MOUNT_POINT/.background/"
# Sets the custom icon volume flag so that volume has nice Orange icon after mount (.VolumeIcon.icns)
/Developer/Tools/SetFile -a C "$MOUNT_POINT"

# Might mot have permissions to do this
{ rm -rf "$MOUNT_POINT/.Trashes/"; } || { echo "Could not remove $MOUNT_POINT/.Trashes/"; }

{ rm -rf "$MOUNT_POINT/.fseventsd/"; } || { echo "Could not remove $MOUNT_POINT/.fseventsd/"; }

hdiutil detach "$DEV_NAME" -force

echo "Converting temporary image to a compressed image."

if [ -e $BUNDLE ]; then
	rm -f $BUNDLE
fi

hdiutil convert $TMP_BUNDLE -format UDZO -imagekey zlib-level=9 -o $BUNDLE

echo "Cleaning up."
rm -f $TMP_BUNDLE
rm -rf $TMP_BUNDLE_DIR

true