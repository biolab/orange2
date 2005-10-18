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
    echo "Usage: ./build.sh <CVS tag> <output filename> <type of build> <version> <compile>"
    exit 1
fi

TAG=$1
OUT=$2
REL=$3
VER=$4
COMPILE=$5

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

rm -rf orange

echo -n "Checkouting Orange from CVS to orange..."
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d orange orange > cvs.log 2>&1
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d orange/source source >> cvs.log 2>&1
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d orange install-scripts/linux/setup.py >> cvs.log 2>&1
cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d orange install-scripts/linux/INSTALL.txt >> cvs.log 2>&1

if [ ! $REL -eq 0 ]; then
    cvs -d :pserver:cvs@estelle.fri.uni-lj.si:/CVS co -r $TAG -f -d orange/OrangeWidgets/Genomics Genomics >> cvs.log 2>&1
fi

echo "done"
# clean CVS co, all CVS directories, all .pyd and .dll files,...

echo -n "Cleaning orange directory..."
find orange -name CVS -type d -exec rm -rf {} \; > /dev/null  2>&1
find orange -name '*.pyd' -type f -exec rm -rf {} \; > /dev/null  2>&1
find orange -name '*.dll' -type f -exec rm -rf {} \; > /dev/null  2>&1
# in every directory create __init__.py file, distutils demand
#find orange -name '*' -type d -exec touch {}/__init__.py \;
# clean everything out of exclude.lst file
cat orange/exclude.lst | xargs rm -rf

if [ $REL -eq 1 ]; then # cleanup of BCMonly.lst file
    cat orange/OrangeWidgets/Genomics/BCMinclude.lst | xargs rm -rf
fi
echo "done"

echo -n "Updating Orange version to Orange-$VER..."
cat orange/setup.py | sed s/"OrangeVer=\"ADDVERSION\""/"OrangeVer=\"Orange-$VER\""/ > orange/new.py
mv -f orange/new.py orange/setup.py
echo "done"

if [ $COMPILE -eq 1 ]; then
	cd orange
        if ! python setup.py compile > ../output.log 2>&1 ; then
              exit 1;
        fi
        cd source 
        mkdir ../tmp_lib
	mv ../*.so ../*.a ../tmp_lib/ 
	make clean
	mv ../tmp_lib/* ..
	rm -rf ../tmp_lib
	cd ../..
fi

echo -n "Packing orange to $OUT..."
tar czpvf $OUT orange > packing.log 2>&1
echo "done"

