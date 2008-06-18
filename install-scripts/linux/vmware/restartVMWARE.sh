#/bin/bash

if [ -a starting ]; then echo already running ; exit ; fi
touch starting

while true; do
	touch starting
	echo
	echo "RESTARTING with clean machine images"
	echo
	cd /home/vmware
	echo "REMOVE OLD IMAGES"
	rm -vRf vmware
	echo
	echo "COPY NEW IMAGES"
	mkdir vmware
	nice cp -vR /vmware/VMWAREimages/* vmware/.
	echo
	echo "EXIT VMWARE TO RESTART WITH CLEAN IMAGES"
	/usr/local/bin/vmware -geometry 1200x1000+100+0
done

