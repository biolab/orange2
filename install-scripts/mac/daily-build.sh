#!/bin/bash -e
#
# Nightly build script that builds and publishes orange dmg installer.
#

function print_usage() {
    echo 'nightly-build.sh [--bootstrap] [--repo HG_REPO] [--template TEMPLATE_URL] PUBLISH_DIR

Note: This script should be run from the root source directory.

Options:

    --bootstrap              Bootstrap the build process (if present this flag must be the
                             in first position)
    -R --repo                HG repository from which to clone/pull the sources in bootstrap mode
    --template TEMPLATE_URL  Path or url to an application template as build
                             by "build-osx-app-template.sh. If not provided
                             a default one will be downloaded.
    -h --help                Print this help

'
}

while [[ ${1:0:1} = "-" ]]; do
    case $1 in
        --bootstrap)
            BOOTSTRAP=1
            WORK_DIR=$(mktemp -d -t orange-daily-build)
            shift 1
            ;;
        -R|--repo)
            REPO=$2
            shift 2
            ;;
        --template)
            TEMPLATE_URL=$2
            shift 2
            ;;
        -*)
            echo "Invalid option $1"
            print_usage
            exit 1
            ;;
    esac
done

REPO=${REPO:-"https://bitbucket.org/biolab/orange"}

PUBLISH_DIR=$1

PUBLISH_DIR=$(cd "$PUBLISH_DIR"; pwd)

TEMPLATE_URL=${TEMPLATE_URL:-"http://orange.biolab.si/download/bundle-templates/Orange.app-template.tar.gz"}

if [[ $BOOTSTRAP ]]; then
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    hg clone "$REPO" orange
    cd orange
    ./install-scripts/mac/daily-build.sh --template "$TEMPLATE_URL" "$PUBLISH_DIR"
    exit 0
fi


VER=$(python setup.py --version)

NODE=$(hg parent --template={node})
REV=$(hg parent --template={rev})

./install-scripts/mac/build-osx-app.sh --template "$TEMPLATE_URL" dist/Orange.app

DMG_NAME=Orange-$VER-$REV.dmg

./install-scripts/mac/create-dmg-installer.sh --app dist/Orange.app dist/$DMG_NAME

echo "Removing old versions"
# Note: project name is prepended to the pattern so the dmg image must
# be named Orange-*.dmg
python setup.py rotate --keep=10 --dist-dir="$PUBLISH_DIR" --match=".dmg"

MD5=$(md5 -q "dist/$DMG_NAME")

echo "Moving dmg installer into place"
echo "==============================="
mv dist/$DMG_NAME "$PUBLISH_DIR"/$DMG_NAME.new

# Check integrity (buggy sshfs??)

MD5_D=$(md5 -q "$PUBLISH_DIR/$DMG_NAME.new")
if [[ $MD5 != $MD5_D ]]; then
    echo "Error moving the bundle in place"
    rm "$PUBLISH_DIR"/$DMG_NAME.new
    exit 1
else
    mv "$PUBLISH_DIR"/$DMG_NAME.new "$PUBLISH_DIR"/$DMG_NAME
fi


echo "Registering new dmg installer"
echo "============================="

if [[ ! -e "$PUBLISH_DIR"/filenames_mac.set ]]; then
    touch "$PUBLISH_DIR"/filenames_mac.set
fi

egrep -v '^(MAC_DAILY)=' "$PUBLISH_DIR"/filenames_mac.set > "$PUBLISH_DIR"/filenames_mac.set.new || true


echo "MAC_DAILY=$DMG_NAME" >> "$PUBLISH_DIR"/filenames_mac.set.new
echo "MAC_DAILY=$DMG_NAME"

mv "$PUBLISH_DIR"/filenames_mac.set.new "$PUBLISH_DIR"/filenames_mac.set

if [[ $BOOTSTRAP ]]; then
    rm -rf $WORK_DIR
fi

echo "$DMG_NAME"
