#!/bin/bash -e
#
# Should be run as: ./wmvare-dailyrun.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'
VMIMAGE='/Users/ailabc/Documents/Virtual Machines.localized/Mac OS X Server 10.5 64-bit.vmwarevm/Mac OS X Server 10.5 64-bit.vmx'
WAIT_TIME=60
IP_ADDRESS='172.16.213.100'

[ $VMRUN list | grep -q "$VMIMAGE" ] && { echo "VMware already running." exit 1; }

$VMRUN start "$VMIMAGE" nogui

# Wait for VMware to start
sleep $WAIT_TIME

# We run it twice so that we als use maybe updated "update-all-scripts.sh" script
ssh ailabc@$IP_ADDRESS /Users/ailabc/update-all-scripts.sh
ssh ailabc@$IP_ADDRESS /Users/ailabc/update-all-scripts.sh

# dailyrun.sh is added to /etc/sudoers so no password is required
# /etc/sudoers entry: ailabc ALL=NOPASSWD:/Users/ailabc/dailyrun.sh
# WARNING: This is generally unsecure as an attacker could change dailyrun.sh file and ...
#          but we are using it in a VMware which is used only for this script, so ...
ssh ailabc@$IP_ADDRESS sudo /Users/ailabc/dailyrun.sh
