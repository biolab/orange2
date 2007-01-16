/home/orange/umount_estelleDownload
/home/orange/mount_estelleDownload

# remove old logs
W=/home/orange/estelleDownload/regressionResults
rm $W/linux.*.log

# copy new ones
cd /home/orange/daily/orange
cp compiling.log $W/linux.compiling.log
cp ../output.log $W/linux.output.log
cp ../regress.log $W/linux.regress.log

cd /home/orange/daily/test_install/orange

for f in *-output; do
	echo copying $f
	mkdir $W/$f
	rm $W/$f/*linux2.*.txt
	cp $f/*crash.txt $f/*error.txt $f/*new.txt $f/*changed.txt $f/*random1.txt $f/*random2.txt $W/$f
done

/home/orange/umount_estelleDownload

