#!/bin/bash -e
#
# Should be run as: ./vmware-status.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'

"$VMRUN" list
