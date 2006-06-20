if ! ps -u vmware | grep Xvnc; then
	echo vncserver not running, starting now
	rm /home/vmware/starting
	/usr/bin/vncserver -geometry 1300x1000
fi
