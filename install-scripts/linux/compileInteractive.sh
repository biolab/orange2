# prepare sources
rm -Rf /home/orange/interactive
mkdir /home/orange/interactive
cd /home/orange/interactive
cvs -Q -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -d orange orange
mkdir orange
mkdir orange/source
cd orange
cvs -Q -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -d source source
cd ..
cp /home/orange/install-scripts/linux/setup.py /home/orange/interactive/orange

# build
cd /home/orange/interactive/orange
VER='1.test'
cat setup.py | sed s/"OrangeVer=\"ADDVERSION\""/"OrangeVer=\"Orange-$VER\""/ > new.py
mv -f new.py setup.py

python setup.py compile
cat compiling.log

