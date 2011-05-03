#!/bin/bash
#
# Matija Polajnar, 28. 7. 2010
# matija.polajnar@fri.uni-lj.si
#
# This script packs the addons and copies them into our official Orange add-on repository.
# An add-on is packed for each directory in ../add-ons. A 'svn up' is issued in each directory
# first, so the contents must be a SVN checkout. 
# The packing only occurs if the addon.txt has been changed!
# TODO: Pack the binary parts too!
#

SCRIPT_DIR=`dirname $0 | xargs readlink -e`
ADDONS_DIR="$SCRIPT_DIR/../add-ons"
ORANGE_DIR="$SCRIPT_DIR/../orange"

# Update the core orange (we need the orngAddOns.py to make a package).
cd "$ORANGE_DIR"
svn up

# For each addon ...
cd "$ADDONS_DIR"
for ADDON in * ; do
  if [[ -d "$ADDONS_DIR/$ADDON" ]] ; then
      echo " ### Processing $ADDON ..."
      cd "$ADDONS_DIR/$ADDON"
      cp addon.xml "../${ADDON}_addon.xml"
      svn revert . -R
      svn up
      if diff addon.xml "../${ADDON}_addon.xml" ; then
        echo "Not changed - not packing!"
      else
        "$SCRIPT_DIR/prepare-and-pack.py" "$ADDONS_DIR/$ADDON" "../${ADDON}.oao"
        echo "put '../${ADDON}.oao'" | sftp addons@biolab.si:/files/
      fi
  fi
done
