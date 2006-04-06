#/bin/bash

while true; do
	echo
	echo "RESTARTING with clean machine images"
	echo
	cd /home/vmware
	echo "REMOVE OLD IMAGES"
	rm -vRf vmware
	echo
	echo "COPY NEW IMAGES"
	mkdir vmware
	cp -vR /vmware/VMWAREimages/* vmware/.
	echo
	echo "EXIT VMWARE TO RESTART WITH CLEAN IMAGES"
	vmware -geometry 1200x1000+100+0
done

