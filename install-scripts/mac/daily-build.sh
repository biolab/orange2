#!/bin/bash -e
#
# Nightly build script that builds and publishes orange dmg installer.
#

function print_usage() {
    echo 'daily-build.sh [--bootstrap] [--repo GIT_REPO] [--template TEMPLATE_URL] PUBLISH_DIR

Note: This script should be run from the root source directory.

Options:

    --bootstrap              Bootstrap the build process (if present this flag must be the
                             in first position)
    -R --repo                GIT repository from which to clone/pull the sources in bootstrap mode
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
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Invalid option $1"
            print_usage
            exit 1
            ;;
    esac
done

REPO=${REPO:-"https://github.com/biolab/orange.git"}

PUBLISH_DIR=$1

PUBLISH_DIR=$(cd "$PUBLISH_DIR"; pwd)

TEMPLATE_URL=${TEMPLATE_URL:-"http://orange.biolab.si/download/files/bundle-templates/Orange.app-template.tar.gz"}

if [[ $BOOTSTRAP ]]; then
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    git clone "$REPO" orange
    cd orange
    ./install-scripts/mac/daily-build.sh --template "$TEMPLATE_URL" "$PUBLISH_DIR"
    exit 0
fi

VER=$(python setup.py --version)

REV=$(git rev-parse --short HEAD)

./install-scripts/mac/build-osx-app.sh --template "$TEMPLATE_URL" dist/Orange.app

DMG_NAME=Orange-$VER-$REV.dmg

./install-scripts/mac/create-dmg-installer.sh --app dist/Orange.app dist/$DMG_NAME


echo "Removing old versions"
# Note: project name is prepended to the pattern so the dmg image must
# be named Orange-*.dmg
python setup.py rotate --keep=10 --dist-dir="$PUBLISH_DIR" --match=".dmg"

./install-scripts/mac/build-publish.sh \
    --dist-dir "$PUBLISH_DIR" \
    --key MAC_DAILY \
    "dist/$DMG_NAME"

if [[ $BOOTSTRAP ]]; then
    rm -rf $WORK_DIR
fi

echo "$DMG_NAME"
