#!/bin/bash
##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs login

if [ $# -lt 2 ]; then
        echo "parameters not given: version RPMfile CVStag"
        exit 1
fi

VER=$1
DMGFILE=$2
TAG=${3:-stable}

echo Version: $VER
echo DMGFILE: $DMGFILE
echo TAG: $TAG
echo

COMPILEORANGE=0
COMPILECRS=0
GEN=0

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
rm -Rf orange doc
cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f orange

if [ $GEN == 1 ]; then
  cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/cvs export -r $TAG -f -d orange/OrangeWidgets/Genomics Genomics
fi

rm -R orange/doc

# remove files we don't want in the installation
rm orange/OrangeWidgets/Visualize/OWLinViz.py
rm orange/OrangeWidgets/Visualize/OWLinVizGraph.py
rm orange/OrangeWidgets/Classify/OWCalibratedClassifier.py
rm orange/OrangeWidgets/Data/OWExampleBuilder.py
rm orange/OrangeWidgets/Data/OWSubsetGenerator.py
rm orange/OrangeWidgets/OWLin_Results.py
rm orange/OrangeWidgets/Other/OWITree.py
rm orange/c45.dll


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

## mkdir estelle.mount
##mount_smbfs -W AI //Administrator@estelle.fri.uni-lj.si/wwwUsers estelle.mount
