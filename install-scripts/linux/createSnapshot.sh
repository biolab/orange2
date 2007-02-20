#!/bin/sh

cd /home/orange/distribution
rm -Rf /home/orange/distribution/*

#preparing all neccessary things for snapshot to be created and put to the \\estelle\download directory
if [ $# -lt 6 ]; then
        echo "only $# parameters given"
	echo "Usage: ./createSnapshot <CVS tag> <output filename> <type of build> <daytag> <var name in filenames.set> <compile>"
	exit 1
fi

VARNAME=$5
NEWFILE=$2
COMPILE=$6
/home/orange/mount_estelleDownload
if ! sh /home/orange/install-scripts/linux/createOrangeDist.sh $1 $2 $3 $4 $6 ; then
        cd /home/orange/distribution/orange
        mail -s "Linux: ERROR compiling Orange" tomaz.curk@fri.uni-lj.si,janez.demsar@fri.uni-lj.si < ../output.log
        cat ../output.log
        echo -e "\n\nERROR compiling when creating distribution, see log above"
        /home/orange/umount_estelleDownload
        exit 1;
fi

if [ ! -e $NEWFILE ]; then
	echo "Something went wrong, see log files (new file does not exist)"
        /home/orange/umount_estelleDownload
	exit 1
fi

OLDFILE=`grep $VARNAME\= /home/orange/estelleDownload/filenames.set | awk -F\= '{print $2}' | tr -d '\r'`
grep -v $VARNAME\= /home/orange/estelleDownload/filenames.set > filenames.new.set
echo $VARNAME=$NEWFILE >> filenames.new.set

rm /home/orange/estelleDownload/$OLDFILE
cp $NEWFILE /home/orange/estelleDownload/.
cp filenames.new.set /home/orange/estelleDownload/filenames.set

/home/orange/umount_estelleDownload

