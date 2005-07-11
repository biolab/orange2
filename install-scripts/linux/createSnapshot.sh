#!/bin/sh

#preparing all neccessary things for snapshot to be created and put to the \\estelle\download directory

if [ $# -lt 4 ]; then
	echo "Usage: ./createSnapshot <CVS tag> <output filename> <type of build> <daytag> <compile>"
	exit 1
fi

VARNAME=$5
NEWFILE=$2

COMPILE=$6
/home/orange/mount_estelleDownload
#echo -n "Preparing $2 filename..."
sh /home/orange/install-scripts/linux/createOrangeDist.sh $1 $2 $3 $4 $6
#echo "done!"

if [ ! -e $NEWFILE ]; then
	echo "Something went wrong, see log files (new file does not exist)"
	exit 1
fi

OLDFILE=`grep $VARNAME\= /mnt/estelleDownload/filenames.set | awk -F\= '{print $2}' | tr -d '\r'`
grep -v $VARNAME\= /mnt/estelleDownload/filenames.set > filenames.new.set
echo $VARNAME=$NEWFILE >> filenames.new.set

rm /mnt/estelleDownload/$OLDFILE > cl.err.log.txt 2>&1
cp $NEWFILE /mnt/estelleDownload/
cp filenames.new.set /mnt/estelleDownload/filenames.set

/home/orange/umount_estelleDownload

