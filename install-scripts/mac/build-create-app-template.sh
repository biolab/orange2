#!/bin/bash -e
# Create (build) an Orange application bundle template
#
# example usage:
#
#   $./build-create-app-template.sh $HOME/Orange.app
#


SCRIPT_DIR_NAME=$(dirname "$0")

BUNDLE_LITE=$SCRIPT_DIR_NAME/bundle-lite/Orange.app

APP=$1

if [[ ! $APP ]]; then
	echo "Applicatition path must be specified"
	echo "Usage: ./build-create-app-template.sh ApplicationTemplate"
	exit 1
fi


PYTHON=$APP/Contents/MacOS/python
EASY_INSTALL=$APP/Contents/MacOS/easy_install
PIP=$APP/Contents/MacOS/pip

export MACOSX_DEPLOYMENT_TARGET=10.5

SDK=/Developer/SDKs/MacOSX$MACOSX_DEPLOYMENT_TARGET.sdk

function create_template {
	# Create a minimal .app template with the expected dir structure
	# Info.plist and icons.

	mkdir -p "$APP"
	mkdir -p "$APP"/Contents/MacOS
	mkdir -p "$APP"/Contents/Resources

	# Copy icons and Info.plist
	cp "$BUNDLE_LITE"/Contents/Resources/* "$APP"/Contents/Resources
	cp "$BUNDLE_LITE"/Contents/Info.plist "$APP"/Contents/Info.plist

	#cp $BUNDLE_LITE/Contents/PkgInfo $APP/Contents/PkgInfo

	cat <<-'EOF' > "$APP"/Contents/MacOS/ENV
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

	# Some non framework libraries are put in $FRAMEWORKS_DIR by machlib standalone
	export DYLD_LIBRARY_PATH="$FRAMEWORKS_DIR"${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
EOF

}

function install_python() {
	download_and_extract http://www.python.org/ftp/python/2.7.5/Python-2.7.5.tgz

	cd Python-2.7.5

	# _hashlib import fails with  Symbol not found: _EVP_MD_CTX_md
	# The 10.5 sdk's libssl does not define it (even though it is v 0.9.7)
	patch setup.py -i - <<-'EOF'
		834c834
		<         min_openssl_ver = 0x00907000
		---
		>         min_openssl_ver = 0x00908000
EOF

	./configure --enable-framework="$APP"/Contents/Frameworks \
				--prefix="$APP"/Contents/Resources \
				--with-universal-archs=intel \
				--enable-universalsdk="$SDK"

	make
	make install
	cd ..

	# PythonAppStart will be used for starting the application GUI.
	# This needs to be symlinked here for Desktop services used the app's
	# Info.plist and not the contained Python.app's
	ln -fs ../Frameworks/Python.framework/Resources/Python.app/Contents/MacOS/Python "$APP"/Contents/MacOS/PythonAppStart
	ln -fs ../Frameworks/Python.framework/Resources/Python.app "$APP"/Contents/Resources/Python.app

	cat <<-'EOF' > "$APP"/Contents/MacOS/python
		#!/bin/bash

		DIRNAME=$(dirname "$0")

		# Set the proper env variables
		source "$DIRNAME"/ENV

		exec -a "$0" "$PYTHONEXECUTABLE" "$@"
EOF

	chmod +x "$APP"/Contents/MacOS/python

	$PYTHON -c"import sys"

}

function install_pip() {
	download_and_extract "https://pypi.python.org/packages/source/p/pip/pip-1.3.1.tar.gz"
	cd pip-1.3.1

	$PYTHON setup.py install
	create_shell_start_script pip

	$PIP --version
}

function install_distribute() {
	download_and_extract "https://pypi.python.org/packages/source/d/distribute/distribute-0.6.45.tar.gz"
	cd distribute-0.6.45

	$PYTHON setup.py install
	create_shell_start_script easy_install

	$EASY_INSTALL --version
}

function install_ipython {
	# install with easy_install (does not work with pip)
	$EASY_INSTALL ipython
	create_shell_start_script ipython
}

function install_qt4 {
	download_and_extract "http://download.qt-project.org/archive/qt/4.7/qt-everywhere-opensource-src-4.7.4.tar.gz"
	cd qt-everywhere-opensource-src-4.7.4

	yes yes | ./configure -prefix "$APP"/Contents/Resources/Qt4 \
				-libdir "$APP"/Contents/Frameworks \
				-framework \
				-release \
				-opensource \
				-no-qt3support \
				-arch x86 -arch x86_64 \
				-no-sql-psql \
				-no-sql-ibase \
				-no-sql-mysql \
				-no-sql-odbc \
				-no-sql-sqlite \
				-no-sql-sqlite2 \
				-nomake examples \
				-nomake demos \
				-nomake docs \
				-nomake translations \
				-sdk "$SDK"

	make -j 4
	make install

	# Register plugins.
	cat <<-EOF > "$APP"/Contents/Resources/qt.conf
		[Paths]
		Plugins = Resources/Qt4/plugins
EOF

	# In case the Python executable is invokes directly we also want it to
	# find the plugins.
	cat <<-EOF > "$APP"/Contents/Frameworks/Python.framework/Resources/Python.app/Contents/Resources/qt.conf
		[Paths]
		Plugins = ../../../../../Resources/Qt4/plugins
EOF

}

function install_sip {
	download_and_extract "http://sourceforge.net/projects/pyqt/files/sip/sip-4.14.6/sip-4.14.6.tar.gz"
	cd sip-4.14.6

	$PYTHON configure.py  --arch i386 --arch x86_64 --sdk "$SDK"

	make
	make install

	$PYTHON -c"import sip"

}

function install_pyqt4 {
	download_and_extract "http://sourceforge.net/projects/pyqt/files/PyQt4/PyQt-4.10.1/PyQt-mac-gpl-4.10.1.tar.gz"
	cd PyQt-mac-gpl-4.10.1

	yes yes | $PYTHON configure.py --qmake "$APP"/Contents/Resources/Qt4/bin/qmake

	make
	make install

	$PYTHON -c"import PyQt4.QtGui, PyQt4.QtGui"

}

function install_pyqwt5 {
	download_and_extract "http://sourceforge.net/projects/pyqwt/files/pyqwt5/PyQwt-5.2.0/PyQwt-5.2.0.tar.gz"

	# configure.py fails (with ld: library not found for -lcrt1.10.5.o) trying to
	# build static libraries
	export CPPFLAGS="--shared"

	cd PyQwt-5.2.0/configure

	$PYTHON configure.py -Q ../qwt-5.2 \
						--extra-cflags="-arch i386 -arch x86_64" \
						--extra-cxxflags="-arch i386 -arch x86_64" \
						--extra-lflags="-arch i386 -arch x86_64"
	make
	make install

	unset CPPFLAGS

	$PYTHON -c"import PyQt4.Qwt5"
}

function install_numpy {
	$PIP install numpy

	$PYTHON -c"import numpy"
}

function install_scipy {
	# This is tricky (req gfortran)
	$PIP install scipy

	$PYTHON -c"import scipy"
}

function download_and_extract() {
	# Usage: download_and_extract http://example/source.tar.gz
	#
	# Download the specified .tar source package and extract it in the current dir
	# If the source package is already present only extract it

	URL=$1
	if [[ ! $URL ]]; then
		echo "An url expected"
		exit 1
	fi

	SOURCE_TAR=$(basename "$URL")

	if [[ ! -e $SOURCE_TAR ]]; then
		echo "Downloading $SOURCE_TAR"
		curl --fail -L --max-redirs 3 $URL -o $SOURCE_TAR
	fi
	tar -xzf $SOURCE_TAR
}


function create_shell_start_script() {
	# Usage: create_shell_start_script pip
	#
	# create a start script for the specified script in $APP/Contents/MacOS

	SCRIPT=$1

	cat <<-'EOF' > "$APP"/Contents/MacOS/"$SCRIPT"
		#!/bin/bash

		DIRNAME=$(dirname "$0")
		NAME=$(basename "$0")

		# Set the proper env variables
		source "$DIRNAME"/ENV

		exec -a "$0" "$DIRNAME"/python "$FRAMEWORKS_DIR"/Python.framework/Versions/Current/bin/"$NAME" "$@"
EOF

	chmod +x "$APP"/Contents/MacOS/"$SCRIPT"
}

function cleanup {
	# Cleanup the application bundle by removing unnecesary files.
	find "$APP"/Contents/ \( -name '*~' -or -name '*.bak' -or -name '*.pyc' -or -name '*.pyo' -or -name '*.pyd' \) -exec rm -rf {} ';'

	find "$APP"/Contents/Frameworks/*Qt*.framework -name '*_debug*' -delete
	find "$APP"/Contents/Frameworks/*Qt*.framework -name '*_debug*' -delete

	find "$APP"/Contents/Frameworks/*Qt*.framework -name '*.la' -delete
	find "$APP"/Contents/Frameworks/*Qt*.framework -name '*.a' -delete
	find "$APP"/Contents/Frameworks/*Qt*.framework -name '*.prl' -delete

}

function make_standalone {
	$PIP install macholib
	$PYTHON -m macholib standalone $APP
	yes y | $PIP uninstall altgraph
	yes y | $PIP uninstall macholib
}

create_template

install_python

install_distribute

install_pip

install_numpy

install_scipy

install_qt4

install_sip

install_pyqt4

install_pyqwt5

install_ipython

cleanup

make_standalone
