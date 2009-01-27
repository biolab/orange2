#!/bin/bash -e
#
# Should be run as: sudo ./fink-restore-selections.sh
#

dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
curl http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-selections.list | dpkg --set-selections
apt-get --assume-yes dselect-upgrade
