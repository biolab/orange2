#!/bin/bash -e
#
# Should be run as: ./wmvare-dailyrun.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'
VMIMAGE='/Users/ailabc/Documents/Virtual Machines.localized/Mac OS X Server 10.5 64-bit.vmwarevm/Mac OS X Server 10.5 64-bit.vmx'
WAIT_TIME=60
RETRIES=5
IP_ADDRESS='172.16.213.100'

# We use public/private keys SSH authentication so no need for password

start_vmware() {
	if "$VMRUN" list | grep -q "$VMIMAGE"; then
		echo "VMware is already running."
		exit 1
	fi
	
	"$VMRUN" start "$VMIMAGE" nogui
	
	# Wait for VMware and OS to start
	sleep $WAIT_TIME
}

stop_vmware() {
	# shutdown is added to /etc/sudoers so no password is required
	# /etc/sudoers entry: ailabc ALL=NOPASSWD:/sbin/shutdown -h now
	ssh ailabc@$IP_ADDRESS "sudo /sbin/shutdown -h now > /dev/null"
	
	# Wait for OS to stop
	sleep $WAIT_TIME
	
	# Ignore errors (VMware should already be stopped)
	"$VMRUN" stop "$VMIMAGE" nogui > /dev/null
}

# Check if autologin was successful
for LOGGED_IN in {1..$RETRIES}; do
	if ssh ailabc@$IP_ADDRESS "who | grep -q console"; then
		# Autologin was successful
		break
	fi
	
	stop_vmware
	
	# Wait for VMware to stop
	sleep $WAIT_TIME
	
	start_vmware
done

if ! ssh ailabc@$IP_ADDRESS "who | grep -q console"; then
	# Autologin was not successful after few retries, give up
	echo "Could not autologin."
	exit 2
fi

# We run it twice so that we als use maybe updated "update-all-scripts.sh" script
ssh ailabc@$IP_ADDRESS "/Users/ailabc/update-all-scripts.sh"
ssh ailabc@$IP_ADDRESS "/Users/ailabc/update-all-scripts.sh"

# dailyrun.sh is added to /etc/sudoers so no password is required
# /etc/sudoers entry: ailabc ALL=NOPASSWD:/Users/ailabc/dailyrun.sh
# WARNING: This is generally unsecure as an attacker could change dailyrun.sh file and ...
#          but we are using it in a VMware which is used only for this script, so ...
ssh ailabc@$IP_ADDRESS "sudo /Users/ailabc/dailyrun.sh"

stop_vmware
