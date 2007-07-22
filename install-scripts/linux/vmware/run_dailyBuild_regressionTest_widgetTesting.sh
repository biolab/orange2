DISPLAY=:1
export DISPLAY
cd /home/vmware

if [ -a /home/vmware/daily_run ]; then
   echo already running
   echo "please check..." | mail -s "already running?" tomaz.curk@fri.uni-lj.si
   exit
fi
touch /home/vmware/daily_run

#L=( winXP.dailyBuild winXP.regressionTests winXP.widgetTesting )

# build orange
L=( noreset.winXP.dailyBuild )
for f in "${L[@]}"; do
   ls $f/*LOCK*
   if [ $? -eq 1 ]; then
	echo "########################################################"
	echo "running VM $f" 

	# run it
	echo "\"$f/Windows\ XP\ Professional.vmx\"" > runThis.lst
	# kill after 1 hour
	ulimit -t 7200
	echo "started at"
	date
	/usr/local/bin/vmware -geometry 1200x1000+0+0 -x -q -k runThis.lst
	echo done at
	date
	echo

	# remove temp VM image
	rm -Rf cron.$f
   else
        echo "$f"  | mail -s "write locks on VM" tomaz.curk@fri.uni-lj.si
	rm /home/vmware/daily_run
	exit
   fi
done

L=( winXP.regressionTests winXP.widgetTesting )
for f in "${L[@]}"; do
   ls $f/*LOCK*
   if [ $? -eq 1 ]; then
	echo "########################################################"
	echo "running VM $f" 
	# copy original VM image into a temporary location
	rm -Rf cron.$f
	cp -vR /vmware/orangeBuildTest.VMWAREimage/$f cron.$f

	# run it
	echo "\"cron.$f/Windows\ XP\ Professional.vmx\"" > runThis.lst
	# kill after 1 hour
	ulimit -t 7200
	echo "started at"
	date
	/usr/local/bin/vmware -geometry 1200x1000+0+0 -x -q -k runThis.lst
	echo done at
	date
	echo

	# remove temp VM image
	rm -Rf cron.$f
   else
        echo "$f"  | mail -s "write locks on VM" tomaz.curk@fri.uni-lj.si
	rm /home/vmware/daily_run
	exit
   fi
done

rm /home/vmware/daily_run

