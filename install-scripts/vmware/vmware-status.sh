#!/bin/bash -e
#
# Should be run as: ./vmware-status.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'

# Sets error handler
trap "echo \"Script failed\"" ERR

"$VMRUN" list
