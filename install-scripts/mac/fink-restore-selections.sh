#!/bin/bash -e
#
# Should be run as: sudo ./fink-restore-selections.sh
#

# Sets error handler
trap "echo \"Script failed\"" ERR

((`id -u` == 0)) || { echo "Must run as root user (use sudo)."; exit 1; }

dpkg --get-selections '*' | cut -f 1 | xargs -n 1 -J % echo % purge | dpkg --set-selections
curl http://www.ailab.si/svn/orange/trunk/install-scripts/mac/fink-selections.list | dpkg --set-selections
apt-get --assume-yes dselect-upgrade
