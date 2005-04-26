#!/bin/sh

#    - pyd vedno ven, vse kar je v exclude.lst
#    - parametri skripti: - ime fajla ki ga generiramo
#			  - tag ki ga uporabim za checkout
#			  - tip builda
#    - poskrbeti da se stvari ne prepletajo pri buildu
# 1. snapshot (TAG: HEAD - snapshot, <poljuben tag> - stable ) 
# 2. genomics - vse iz 1. + modul iz CVSja Genomics v orange/OrangeWidgets/Genomics (numarray - error ce ga ni)
#	- kar je v BCMonly.lst ne gre zraven
# 3. BCM - vsi iz 2. + posebni fajli
#	 - vsi fajli ki so zlistani v orange/OrangeWidgets/Genomics/BCMonly.lst

if [ $# -lt 4 ]; then
    echo "Usage: ./build.sh <output file> <CVS tag> <type of build> <version>"
    exit 1
fi

OUT=$1
TAG=$2
REL=$3
VER=$4

# check which kind of package should build, default is 'normal'
if [ $REL == "Genomics" ]; then
    REL=1
elif [ $REL == "genomics" ]; then
    REL=1
elif [ $REL == "GENOMICS" ]; then
    REL=1
elif [ $REL == "BCM" ]; then
    REL=2
elif [ $REL == "Bcm" ]; then
    REL=2
elif [ $REL == "bcm" ]; then
    REL=2
else
    REL=0
fi


if [ $REL -eq 1 ]; then
    ORANGEDIR=Orange-Genomics-$VER
elif [ $REL -eq 2 ]; then
    ORANGEDIR=Orange-BCM-$VER
else
    ORANGEDIR=Orange-$VER
fi

rm -rf $ORANGEDIR

echo -n "Checkouting Orange from CVS to $ORANGEDIR..."
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d $ORANGEDIR orange > cvs.log 2>&1
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d $ORANGEDIR/source source >> cvs.log 2>&1
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d $ORANGEDIR install-scripts/setup.py >> cvs.log 2>&1

if [ ! $REL -eq 0 ]; then
    cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d $ORANGEDIR/OrangeWidgets/Genomics Genomics >> cvs.log 2>&1
fi

echo "done"
# clean CVS co, all CVS directories, all .pyd and .dll files,...

echo -n "Cleaning $ORANGEDIR..."
find $ORANGEDIR -name CVS -type d -exec rm -rf {} \; > /dev/null  2>&1
find $ORANGEDIR -name '*.pyd' -type f -exec rm -rf {} \; > /dev/null  2>&1
find $ORANGEDIR -name '*.dll' -type f -exec rm -rf {} \; > /dev/null  2>&1
# in every directory create __init__.py file, distutils demand
find $ORANGEDIR -name '*' -type d -exec touch {}/__init__.py \;
# clean everything out of exclude.lst file
cat $ORANGEDIR/exclude.lst | xargs rm -rf

if [ $REL -eq 1 ]; then # cleanup of BCMonly.lst file
    cat $ORANGEDIR/OrangeWidgets/Genomics/BCMonly.lst | xargs rm -rf
fi
echo "done"

echo -n "Updating Orange version to Orange-$VER..."
cat $ORANGEDIR/setup.py | sed s/"OrangeVer=\"ADDVERSION\""/"OrangeVer=\"Orange-$VER\""/ > $ORANGEDIR/new.py
mv -f $ORANGEDIR/new.py $ORANGEDIR/setup.py
echo "done"

echo -n "Packing $ORANGEDIR to $OUT..."
tar czpvf $OUT $ORANGEDIR > packing.log 2>&1
echo "done"
#python setup.py install
