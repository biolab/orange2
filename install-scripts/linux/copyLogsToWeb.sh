/home/orange/umount_estelleDownload
/home/orange/mount_estelleDownload

# remove old logs
W=/home/orange/estelleDownload/regressionResults/linux
rm $W/*.log

# copy new ones
cd /home/orange/daily/orange
cp -p *.log ../*.log $W

cd /home/orange/daily/test_install/orange
rm -Rf $W/tests
mkdir $W/tests

for f in *-output; do
	echo copying $f
	mkdir $W/tests/$f
	cp -p $f/*.txt $W/tests/$f
done

cp testresults.xml $W/tests

/home/orange/umount_estelleDownload

