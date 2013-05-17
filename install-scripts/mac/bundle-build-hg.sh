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
TEMPLATE_VERSION=`curl --silent http://orange.biolab.si/download/bundle-templates/CURRENT.txt`
curl --silent http://orange.biolab.si/download/bundle-templates/Orange-template-${TEMPLATE_VERSION}.tar.gz | tar -xz -C $WORK_DIR

# Make repos dir if it does not yet exist
if [ ! -e $REPOS_DIR ]; then
	mkdir $REPOS_DIR
fi

# Create bundle startup script
cat <<-'EOF' > ${TMP_BUNDLE_DIR}/Orange.app/Contents/MacOS/Orange
	#!/bin/bash

	source `dirname "$0"`/ENV

	# LaunchServices passes the Carbon process identifier to the application with
	# -psn parameter - we do not want it
	if [[ $1 == -psn_* ]]; then
	    shift 1
	fi

	exec -a "$0" "$PYTHONEXECUTABLE" -m Orange.OrangeCanvas.main "$@"
EOF

chmod +x ${TMP_BUNDLE_DIR}/Orange.app/Contents/MacOS/Orange

# Python interpreter in the bundle
PYTHON=${TMP_BUNDLE_DIR}/Orange.app/Contents/MacOS/python

# easy_install script in the bundle
EASY_INSTALL=${TMP_BUNDLE_DIR}/Orange.app/Contents/MacOS/easy_install

#Python version
PY_VER=`$PYTHON -c "import sys; print sys.version[:3]"`

# First install/upgrade distrubute. The setup.py scripts might
# need it
echo "Installing/upgrading distribute in the bundle"
echo "============================================="
$EASY_INSTALL -U distribute


echo "Checkouting and building orange"
echo "==============================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange orange $REVISION $REPOS_DIR ${TMP_BUNDLE_DIR}/Orange.app

echo "Specifically building orangeqt"
echo "------------------------------"

CUR_DIR=`pwd`
cd $REPOS_DIR/orange/source/orangeqt
echo "Fixing sip/pyqt configuration"

APP=${TMP_BUNDLE_DIR}/Orange.app
APP_ESCAPED=`echo ${TMP_BUNDLE_DIR}/Orange.app | sed s/'\/'/'\\\\\/'/g`
sed -i.bak "s/Users.*Orange.app/$APP_ESCAPED/g"  $APP/Contents/Frameworks/Python.framework/Versions/$PY_VER/lib/python$PY_VER/site-packages/PyQt4/pyqtconfig.py
sed -i.bak "s/Users.*Orange.app/$APP_ESCAPED/g"  $APP/Contents/Frameworks/Python.framework/Versions/$PY_VER/lib/python$PY_VER/site-packages/sipconfig.py
export PATH=$APP/Contents/Resources/Qt4/bin:$PATH
$PYTHON setup.py install
cd $CUR_DIR

echo "Fixing Qt plugins search path"
echo "[Paths]
Plugins = ../../../../../Resources/Qt4/plugins/" > $APP/Contents/Frameworks/Python.framework/Resources/Python.app/Contents/Resources/qt.conf


echo "Checkouting and building bioinformatics addon"
echo "============================================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange-bioinformatics bioinformatics tip $REPOS_DIR ${TMP_BUNDLE_DIR}/Orange.app

echo "Checkouting and building text addon"
echo "==================================="
./bundle-inject-hg.sh https://bitbucket.org/biolab/orange-text text tip $REPOS_DIR ${TMP_BUNDLE_DIR}/Orange.app

echo "Installing networkx"
echo "+++++++++++++++++++"
./bundle-inject-pypi.sh networkx-1.6 http://pypi.python.org/packages/source/n/networkx/networkx-1.6.tar.gz $REPOS_DIR ${TMP_BUNDLE_DIR}/Orange.app

echo "Installing suds library"
echo "+++++++++++++++++++++++"
./bundle-inject-pypi.sh suds-0.4 http://pypi.python.org/packages/source/s/suds/suds-0.4.tar.gz $REPOS_DIR ${TMP_BUNDLE_DIR}/Orange.app

echo "Instaling slumber library"
echo "+++++++++++++++++++++++++"
$EASY_INSTALL slumber

echo "Removing unnecessary files."
find $TMP_BUNDLE_DIR \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' -or -name '*.pyd' \) -exec rm -rf {} ';'

ln -s ../Frameworks/Python.framework/Versions/Current/lib/python${PY_VER}/site-packages/Orange ${TMP_BUNDLE_DIR}/Orange.app/Contents/Resources/Orange

	
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