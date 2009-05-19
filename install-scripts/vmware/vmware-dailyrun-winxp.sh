#!/bin/bash -e
#
# Should be run as: ./wmvare-dailyrun-winxp.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'
VMIMAGE='/Users/ailabc/Documents/Virtual Machines.localized/winXP.dailyBuild/Windows XP Professional.vmx'
WAIT_TIME=3600
NAME='Windows XP'

start_vmware() {
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "[$NAME] VMware is already running."
		exit 1
	fi
	
	"$VMRUN" start "$VMIMAGE" nogui

	return 0
}

stop_vmware() {
	# Wait for VMware and OS to start
	sleep $WAIT_TIME
	
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "[$NAME] Had to force shutdown."
		"$VMRUN" stop "$VMIMAGE" nogui
	fi
	
	return 0
}

start_vmware
#stop_vmware
