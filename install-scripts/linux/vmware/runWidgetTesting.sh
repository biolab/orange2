DISPLAY=george:1
export DISPLAY
cd /home/vmware
rm -Rf testWidgetImage
cp -R /vmware/VMWAREimages/winXP.widgetTesting testWidgetImage
echo "\"testWidgetImage/Windows\ XP\ Professional.vmx\"" > runThis.lst

# kill after 1 hour
ulimit -t 7200
/usr/local/bin/vmware -geometry 1200x1000+0+0 -x -q -k runThis.lst
date
echo done

