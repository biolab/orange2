#!/bin/bash -e

umount /Volumes/fink/
mkdir -p /Volumes/fink/
/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) fink@biolab.si:/files /Volumes/fink/

umount /Volumes/download/
mkdir -p /Volumes/download/
/Users/ailabc/Downloads/sshfs-binaries/sshfs-static-leopard -o reconnect,workaround=nonodelay,uid=$(id -u),gid=$(id -g) download@biolab.si:/files /Volumes/download/
