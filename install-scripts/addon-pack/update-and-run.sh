#!/bin/bash
#
# Matija Polajnar, 28. 7. 2010
# matija.polajnar@fri.uni-lj.si
#
# This script issues an update of the packaging scripts and runs the packaging. 
#

(
  echo "=== `date` ==="
  sleep 120
  cd `dirname $0`
  ./update-scripts.sh
  ./pack-addons.sh
  poweroff
) >> /var/log/update-and-run.log
