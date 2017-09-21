#!/bin/bash -e
# Build an OSX Applicaiton (.app) for Orange Canvas
#
# Example:
#
#     $ build-osx-app.sh $HOME/Applications/Orange.app
#

function print_usage() {
    echo 'build-osx-app.sh [-i] [--template] Orange.app

Build an Orange Canvas OSX application bundle (Orange.app).

NOTE: this script should be run from the source root directory.

Options:

    --template TEMPLATE_URL  Path or url to an application template as build
                             by "build-osx-app-template.sh. If not provided
                             a default one will be downloaded.
    -i --inplace             The provided target application path is already
                             a template into which Orange should be installed
                             (this flag cannot be combined with --template).
    -h --help                Print this help'
}

while [[ ${1:0:1} = "-" ]]; do
    case $1 in
        --template)
            TEMPLATE_URL=$2
            shift 2;
            ;;
        -i|--inplace)
            INPLACE=1
            shift 1
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Unknown argument $1"
            print_usage
            exit 1
            ;;
    esac
done

# extended glob expansion / fail on filename expansion
shopt -s extglob failglob

if [[ ! -f setup.py ]]; then
    echo "This script must be run from the source root directory!"
    print_usage
    exit 1
fi

APP=${1:-dist/Orange.app}

if [[ $INPLACE ]]; then
    if [[ $TEMPLATE_URL ]]; then
        echo "--inplace and --template can not be combined"
        print_usage
        exit 1
    fi

    if [[ -e $APP && ! -d $APP ]]; then
        echo "$APP exists and is not a directory"
        print_usage
        exit 1
    fi
fi

TEMPLATE_URL=${TEMPLATE_URL:-"http://orange.biolab.si/download/files/bundle-templates/Orange.app-template.tar.gz"}

SCHEMA_REGEX='^(https?|ftp|local)://.*'

if [[ ! $INPLACE ]]; then
    BUILD_DIR=$(mktemp -d -t orange-build)

    echo "Retrieving a template from $TEMPLATE_URL"
    # check for a url schema
    if [[ $TEMPLATE_URL =~ $SCHEMA_REGEX ]]; then
        curl --fail --silent -L --max-redirs 2 "$TEMPLATE_URL" | tar -x -C "$BUILD_DIR"
        TEMPLATE=( $BUILD_DIR/*.app )

    elif [[ -d $TEMPLATE_URL ]]; then
        cp -a $TEMPLATE_URL $BUILD_DIR
        TEMPLATE=$BUILD_DIR/$(basename "$TEMPLATE_URL")

    elif [[ -e $TEMPLATE_URL ]]; then
        # Assumed to be an archive
        tar -xf "$TEMPLATE_URL" -C "$BUILD_DIR"
        TEMPLATE=( $BUILD_DIR/*.app )
    else
        echo "Invalid --template $TEMPLATE_URL"
        exit 1
    fi
else
    TEMPLATE=$APP
fi
echo "Building application in $TEMPLATE"

PYTHON=$TEMPLATE/Contents/MacOS/python
PIP=$TEMPLATE/Contents/MacOS/pip

PREFIX=$("$PYTHON" -c'import sys; print sys.prefix')
SITE_PACKAGES=$("$PYTHON" -c'import sysconfig as sc; print(sc.get_path("platlib"))')

echo "Building Orange"
echo "==============="
"$PYTHON" setup.py build

echo "Building orangeqt extension"
echo "==========================="

FDIR=$TEMPLATE/Contents/Frameworks
# to find moc executable in the app bundle
EXTRA_PATH=$PREFIX/bin:$TEMPLATE/Contents/Resources/Qt4/bin
# for the compiler to find Qt's headers and frameworks
EXTRA_CXXFLAGS="-F$FDIR -I$FDIR/QtCore.framework/Headers -I$FDIR/QtGui.framework/Headers"
EXTRA_LDFLAGS="-F$FDIR -framework QtCore -framework QtGui"

echo "Fixing sip/pyqt configuration"

sed -i.bak "s@/.*/Orange.app/@$TEMPLATE/@g" "${SITE_PACKAGES}"/PyQt4/pyqtconfig.py
sed -i.bak "s@/.*/Orange.app/@$TEMPLATE/@g" "${SITE_PACKAGES}"/sipconfig.py


(
    PATH=$EXTRA_PATH:$PATH
    CXXFLAGS=${EXTRA_CXXFLAGS}${CXXFLAGS:+:$CXXFLAGS}
    LDFLAGS=${EXTRA_LDFLAGS}:${LDFLAGS:+:$LDFLAGS}
    "$PYTHON" setup.py build_pyqt_ext
)

echo "Installing Orange"
echo "================="

"$PYTHON" setup.py install

cat <<-'EOF' > "$TEMPLATE"/Contents/MacOS/Orange
	#!/bin/bash

	unset -v PYTHONPATH

	DIRNAME=$(dirname "$0")
	source "$DIRNAME"/ENV

	# LaunchServices passes the Carbon process identifier to the application with
	# -psn parameter - we do not want it
	if [[ $1 == -psn_* ]]; then
	    shift 1
	fi

	exec -a "$0" "$DIRNAME"/PythonAppStart -m Orange.OrangeCanvas.main "$@"
EOF

chmod +x "$TEMPLATE"/Contents/MacOS/Orange


VERSION=$(python setup.py --version)
cat <<EOF > "$TEMPLATE"/Contents/Info.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>NSPrincipalClass</key>
	<string>NSApplication</string>
	<key>NSHighResolutionCapable</key>
	<true/>
	<key>CFBundleExecutable</key>
	<string>Orange</string>
	<key>CFBundleIconFile</key>
	<string>orange.icns</string>
	<key>CFBundleIdentifier</key>
	<string>si.biolab.Orange</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>Orange</string>
	<key>CFBundleGetInfoString</key>
	<string>Orange, component-based data mining software</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleSignature</key>
	<string>Orng</string>
	<key>CFBundleShortVersionString</key>
	<string>${VERSION}</string>
	<key>CFBundleVersion</key>
	<string>${VERSION}</string>
	<key>CFBundleDocumentTypes</key>
	<array>
		<dict>
			<key>CFBundleTypeExtensions</key>
			<array>
				<string>ows</string>
			</array>
			<key>CFBundleTypeName</key>
			<string>Orange Canvas Schema</string>
			<key>CFBundleTypeOSTypes</key>
			<array>
				<string>OWSf</string>
			</array>
			<key>CFBundleTypeIconFile</key>
			<string>schema.icns</string>
			<key>CFBundleTypeRole</key>
			<string>Viewer</string>
			<key>LSIsAppleDefaultForType</key>
			<true/>
		</dict>
	</array>
</dict>
</plist>
EOF
echo "Installing add ons"
echo "=================="

# Install as eggs. The application's Add-ons dialog has problem with
# upgrading flat installs.
"$PIP" install --egg orange-bioinformatics
"$PIP" install --egg orange-network
"$PIP" install --egg hg+https://bitbucket.org/biolab/orange-text


echo "Installing optional dependencies"
echo "================================"
"$PIP" install slumber


if [[ ! $INPLACE ]]; then
    echo "Moving the application to $APP"
    if [[ -e $APP ]]; then
        rm -rf "$APP"
    fi
    mkdir -p $(dirname "$APP")
    mv "$TEMPLATE" "$APP"
fi
