#!/bin/bash -e

if [ -e /Library/Filesystems/fusefs.fs/Support/fusefs.kext ]; then
	kextload -quiet /Library/Filesystems/fusefs.fs/Support/fusefs.kext
fi

umount /Volumes/fink/ 2> /dev/null || true
sleep 5
mkdir -p /Volumes/fink/
/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) fink@biolab.si:/files /Volumes/fink/

umount /Volumes/download/ 2> /dev/null || true
sleep 5
mkdir -p /Volumes/download/
/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) download@biolab.si:/files /Volumes/download/

sleep 5
