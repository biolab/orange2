#!/bin/bash
cd ~/install-scripts/mac

##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs login

if [ $# -ne 5 ]; then
        echo "parameters not given: CVStag DMGfile VARname INCLUDEgenomics COMPILEORANGE"
        exit 1
fi

TAG=$1
DMGFILE=$2
VARNAME=$3
INCGENOMICS=$4
COMPILEORANGE=$5

echo Tag: $TAG
echo DMGFILE: $DMGFILE
echo VARNAME: $VARNAME
echo INCGENOMICS: $INCGENOMICS
echo COMPILE ORANGE: $COMPILEORANGE
echo
gcc -v
echo

# COMPILEORANGE=1
COMPILECRS=0

## check out orange source and compile orange
mkdir compiledOrange
if [ $COMPILEORANGE == 1 ]; then
  rm -Rf source
  cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f source
  cd source
  python makedep.py
  if ! make -f Makefile.mac; then
    exit 1
  fi
  cd ..
fi
mv *.so compiledOrange


## compile orngCRS
if [ $COMPILECRS == 1 ]; then
  rm -Rf orngExtn-1_8_1_py23
  tar -xvzf orngExtn-1_8_1_py23.mac.tar.gz
  cd orngExtn-1_8_1_py23
  python setup.py build
  mv build/lib.darwin-6.8-Power_Macintosh-2.3/_orngCRS.so ../compiledOrange
  cd ..
fi

## check out orange modules
rm -Rf orange
cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f orange

if [ $INCGENOMICS == 1 ]; then
  cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f -d orange/OrangeWidgets/Genomics Genomics
fi

rm -R orange/doc

# remove files we don't want in the installation
for f in `cat orange/exclude.lst`; do
  echo removing $f
  rm $f
done

rm orange/*.pyd ## no need for windows files

## after compiling orange, move it out of the path and into the orange directory
cp compiledOrange/*.so orange
PYTHONPATH=.
rm -Rf build
if ! python makeapplication.py --resource=orange build; then
  exit 1
fi

## create image file and copy the compiled application into it
rm tmp.dmg
# just in case, if already mounted
hdiutil unmount /Volumes/Orange
hdiutil create -size 64m -type UDIF -fs HFS+ -volname Orange tmp.dmg
hdiutil mount tmp.dmg
cp -R build/Orange.app /Volumes/Orange
hdiutil unmount /Volumes/Orange
## hdiutil resize tmp.dmg -size min
hdiutil convert -format UDZO tmp.dmg -o $DMGFILE
## hdiutil unmount tmp.dmg
rm tmp.dmg

## copy file to estelle and change version
~/mount_estelle

# remember name of old file
OLDDMGFILE=`grep $VARNAME\= ~/estelleDownload/filenames.set | awk -F\= '{print $2}' | tr -d '\r'`
# change name to new filename
grep -v $VARNAME\= ~/estelleDownload/filenames.set > filenames.new.set
echo $VARNAME=$DMGFILE >> filenames.new.set

# first remove old file (in case same name as new)
rm ~/estelleDownload/$OLDDMGFILE $> cl.err.log.txt
cp $DMGFILE ~/estelleDownload
cp filenames.new.set ~/estelleDownload/filenames.set
# remove old file
rm *.dmg

~/umount_estelle

