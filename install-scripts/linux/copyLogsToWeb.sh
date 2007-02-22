/home/orange/umount_estelleDownload
/home/orange/mount_estelleDownload

# remove old logs
W=/home/orange/estelleDownload/regressionResults
rm $W/linux.*.log

# copy new ones
if cd /home/orange/daily/orange ; then
	cp compiling.log $W/linux.compiling.log
	cp install.log $W/linux.install.log
	cp regress.log $W/linux.regress.log
	cp ../output.log $W/linux.output.log
else
	echo "no log files"
fi

if cd /home/orange/daily/test_install/orange ; then
	for f in *-output; do
		echo copying $f
		mkdir $W/regressionTests/$f
		rm $W/regressionTests/$f/*linux2.*.txt
		cp $f/*crash.txt $f/*error.txt $f/*new.txt $f/*changed.txt $f/*random1.txt $f/*random2.txt $W/regressionTests/$f
	done
else
	echo "regression test were not run"
fi

/home/orange/umount_estelleDownload

