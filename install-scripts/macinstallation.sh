##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs login


## compile orange
##rm -Rf source
##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs export -r HEAD source
##cd source
##python makedep.py
##make -f Makefile.mac
##cd ..
mv *.so compiledOrange


## check out orange source and modules
rm -Rf orange doc
cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs export -r HEAD orange
mv orange/doc .


## after compiling orange, move it out of the path and into the orange directory
cp compiledOrange/*.so orange
setenv PYTHONPATH .
rm -Rf build
python makeapplication.py --resource=orange build
cp emptyDiskImages/* ~/Desktop
open build
open ~/Desktop/orange.dmg
echo "drag orange application into orange mount drive"
echo "resize and convert with the Disk Copy utility"
open /Applications/Utilities/Disk\ Copy.app
cat

open . 
open ~/Desktop/orangeDoc.dmg
echo "drag documentation into orange documentation mount drive"
echo "resize and convert with the Disk Copy utility"
open /Applications/Utilities/Disk\ Copy.app

