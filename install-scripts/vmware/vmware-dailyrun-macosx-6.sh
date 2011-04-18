#!/bin/bash
#
# Should be run as: ./wmvare-dailyrun-macosx-6.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'
VMIMAGE='/Users/ailabc/Documents/Virtual Machines.localized/Mac OS X Server 10.6 64-bit.vmwarevm/Mac OS X Server 10.6 64-bit.vmx'
WAIT_START_TIME=300
WAIT_STOP_TIME=600
WAIT_RESTART_TIME=120
RETRIES=5
IP_ADDRESS='172.16.213.101'
NAME='Mac OS X 10.6'

# We use public/private keys SSH authentication so no need for a password

start_vmware() {
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "[$NAME] VMware is already running."
		exit 1
	fi
	
	# We hide some Mac OS X warnings which happen if nobody is logged into a host Mac OS X
	"$VMRUN" start "$VMIMAGE" nogui 2>&1 | grep -i -v 'Untrusted apps are not allowed to connect to or launch Window Server before login' | grep -i -v 'FAILED TO establish the default connection to the WindowServer' | grep -i -v 'kCGErrorFailure'
	ps=("${PIPESTATUS[@]}")
	# PIPESTATUS check is needed so that we test return value of the VMRUN and not grep
	if ((${ps[0]})); then
		echo "[$NAME] Could not start VMware."
		exit 2
	fi
	
	# Wait for VMware and OS to start
	sleep $WAIT_START_TIME
	
	return 0
}

stop_vmware() {
	# shutdown is added to /etc/sudoers so no password is required
	# /etc/sudoers entry: ailabc ALL=NOPASSWD:/sbin/shutdown -h now
	ssh ailabc@$IP_ADDRESS "sudo /sbin/shutdown -h now > /dev/null"
	
	# Wait for OS to stop
	sleep $WAIT_STOP_TIME
	
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "[$NAME] Have to force shutdown."
		# We hide some Mac OS X warnings which happen if nobody is logged into a host Mac OS X
		"$VMRUN" stop "$VMIMAGE" nogui 2>&1 | grep -i -v 'Untrusted apps are not allowed to connect to or launch Window Server before login' | grep -i -v 'FAILED TO establish the default connection to the WindowServer' | grep -i -v 'kCGErrorFailure' | true
		ps=("${PIPESTATUS[@]}")
		# PIPESTATUS check is needed so that we test return value of the VMRUN and not grep
		if ((${ps[0]})); then
			echo "[$NAME] Could not stop VMware."
			exit 3
		fi
	fi
	
	return 0
}

start_vmware

# Check if autologin was successful
for LOGGED_IN in {1..$RETRIES}; do
	if ssh ailabc@$IP_ADDRESS "who | grep -q console"; then
		# Autologin was successful
		break
	fi
	
	stop_vmware
	
	# Wait for VMware to stop
	sleep $WAIT_RESTART_TIME
	
	start_vmware
done

if ! ssh ailabc@$IP_ADDRESS "who | grep -q console"; then
	# Autologin was not successful after few retries, give up
	echo "[$NAME] Could not autologin."
	stop_vmware
	exit 4
fi

# We run it twice so that we also use maybe updated "update-all-scripts.sh" script
ssh ailabc@$IP_ADDRESS "/Users/ailabc/update-all-scripts.sh"
ssh ailabc@$IP_ADDRESS "/Users/ailabc/update-all-scripts.sh"

# dailyrun-finkonly-withsource.sh is added to /etc/sudoers so no password is required
# /etc/sudoers entry: ailabc ALL=NOPASSWD:/Users/ailabc/dailyrun-finkonly-withsource.sh
# WARNING: This is generally insecure as an attacker could change dailyrun-finkonly-withsource.sh file and ...
#          but we are using it in a VMware which is used only for this script, so ...
ssh ailabc@$IP_ADDRESS "sudo /Users/ailabc/dailyrun-finkonly-withsource.sh"

stop_vmware
