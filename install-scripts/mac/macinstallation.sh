#!/bin/bash
##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs login

if [ $# -ne 4 ]; then
        echo "parameters not given: CVStag DMGfile VARname INCLUDEgenomics"
        exit 1
fi

TAG=$1
DMGFILE=$2
VARNAME=$3
INCGENOMICS=$4

echo Tag: $TAG
echo DMGFILE: $DMGFILE
echo VARNAME: $VARNAME
echo INCGENOMICS: $INCGENOMICS
echo

COMPILEORANGE=0
COMPILECRS=0

## check out orange source and compile orange
mkdir compiledOrange
if [ $COMPILEORANGE == 1 ]; then
  rm -Rf source
  cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f source
  cd source
  python makedep.py
  make -f Makefile.mac
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
  echo removing orange/$f
  rm orange/$f
done

## rm orange/OrangeWidgets/Visualize/OWLinVizGraph.py
## rm orange/c45.dll


## after compiling orange, move it out of the path and into the orange directory
cp compiledOrange/*.so orange
PYTHONPATH=.
rm -Rf build
python makeapplication.py --resource=orange build

## create image file and copy the compiled application into it
rm tmp.dmg
hdiutil create -size 64m -type UDIF -fs HFS+ -volname Orange tmp.dmg
hdiutil mount tmp.dmg
cp -R build/Orange.app /Volumes/Orange
hdiutil unmount /Volumes/Orange
## hdiutil resize tmp.dmg -size min
hdiutil convert -format UDZO tmp.dmg -o $DMGFILE
rm tmp.dmg

## copy file to estelle and change version
~/mount_estelle
# remember name of old file
OLDDMGFILE=`grep MACINTOSH_SNAPSHOT ~/estelleDownload/filenames.set | awk -F\= '{print $2}'`
# change name to new filename
grep -v $VARNAME ~/estelleDownload/filenames.set > filenames.new.set
echo $VARNAME=$DMGFILE >> filenames.new.set
cp $DMGFILE ~/estelleDownload
cp filenames.new.set ~/estelleDownload/filenames.set
# remove old file
rm ~/estelleDownload/$OLDDMGFILE
# ~/umount_estelle

