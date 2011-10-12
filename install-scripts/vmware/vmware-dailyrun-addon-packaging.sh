#!/bin/bash -e
#
# Should be run as: ./wmvare-dailyrun-winxp.sh
#

VMRUN='/Library/Application Support/VMware Fusion/vmrun'
VMIMAGE='/Volumes/Data/vmware/Orange Add-on Packaging/Orange Add-on Packaging.vmx'
WAIT_START_TIME=300
WAIT_STOP_TIME=600
IP_ADDRESS='172.16.213.134'
NAME='Orange Add-on Packaging'

# We use public/private keys SSH authentication so no need for a password

start_vmware() {
    if "$VMRUN" list | grep -q "$VMIMAGE"; then
        echo "[$NAME] VMware is already running."
        exit 1
    fi
    
    # We hide some Mac OS X warnings which happen if nobody is logged into a host Mac OS X
    "$VMRUN" start "$VMIMAGE" nogui 2>&1 | grep -i -v 'Untrusted apps are not allowed to connect to or launch Window Server before login' | grep -i -v 'FAILED TO establish the default connection to the WindowServer' | grep -i -v 'kCGErrorFailure' || true
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
    ssh root@$IP_ADDRESS "/sbin/shutdown -h now > /dev/null"
    
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

ssh root@$IP_ADDRESS "/root/update-and-run.sh"

stop_vmware