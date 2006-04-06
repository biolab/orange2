if ! ps -u vmware | grep vmware; then
	echo vncserver not running, starting now
	vncserver -geometry 1300x1000
fi
