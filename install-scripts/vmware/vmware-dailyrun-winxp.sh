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
	
	# We hide some Mac OS X warnings which happen if nobody is logged into a host Mac OS X
	"$VMRUN" start "$VMIMAGE" nogui 2>&1 | grep -i -v 'Untrusted apps are not allowed to connect to or launch Window Server before login' | grep -i -v 'FAILED TO establish the default connection to the WindowServer' | true
	# PIPESTATUS check is needed so that we test return value of the VMRUN and not grep
	# (which would be otherwise checked because of the -e switch, with true at the end of the pipe we ignore it)
	if ((${PIPESTATUS[0]})); then false; fi

	return 0
}

stop_vmware() {
	# Wait for VMware and OS to start
	sleep $WAIT_TIME
	
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "[$NAME] Had to force shutdown."
		# We hide some Mac OS X warnings which happen if nobody is logged into a host Mac OS X
		"$VMRUN" stop "$VMIMAGE" nogui 2>&1 | grep -i -v 'Untrusted apps are not allowed to connect to or launch Window Server before login' | grep -i -v 'FAILED TO establish the default connection to the WindowServer' | true
		# PIPESTATUS check is needed so that we test return value of the VMRUN and not grep
		# (which would be otherwise checked because of the -e switch, with true at the end of the pipe we ignore it)
		if ((${PIPESTATUS[0]})); then false; fi
	fi
	
	return 0
}

start_vmware
#stop_vmware
