##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs login


## check out orange source and compile orange
mkdir compiledOrange
##rm -Rf source
##cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs export -r HEAD source
##cd source
##python makedep.py
##make -f Makefile.mac
##cd ..
mv *.so compiledOrange


## compile orngCRS
##rm -Rf orngExtn-1_8_1_py23
##tar -xvzf orngExtn-1_8_1_py23.mac.tar.gz
##cd orngExtn-1_8_1_py23
##python setup.py build
##mv build/lib.darwin-6.8-Power_Macintosh-2.3/_orngCRS.so ../compiledOrange
##cd ..


## check out orange modules
rm -Rf orange doc
cvs -d :pserver:tomazc@estelle.fri.uni-lj.si:/d//cvs export -r HEAD orange
mv orange/doc Orange\ Doc

# remove files we don't want in the installation
rm orange/OrangeWidgets/Visualize/OWLinViz.py
rm orange/OrangeWidgets/Visualize/OWLinVizGraph.py
rm orange/OrangeWidgets/Classify/OWCalibratedClassifier.py
rm orange/OrangeWidgets/Data/OWExampleBuilder.py
rm orange/OrangeWidgets/Data/OWSubsetGenerator.py
rm orange/OrangeWidgets/OWLin_Results.py
rm orange/OrangeWidgets/Other/OWITree.py
rm -Rf orange/OrangeWidgets/Genomics

## after compiling orange, move it out of the path and into the orange directory
cp compiledOrange/*.so orange
setenv PYTHONPATH .
rm -Rf build
python makeapplication.py --resource=orange build
tar -C ~/Desktop -xvzf emptyDiskImages/Orange.tar.gz
open build
open ~/Desktop/Orange.dmg
echo "drag orange application into orange mount drive"
echo "resize and convert with the Disk Copy utility"
open /Applications/Utilities/Disk\ Copy.app
cat

tar -C ~/Desktop -xvzf emptyDiskImages/OrangeDoc.tar.gz
open . 
open ~/Desktop/Orange\ Doc.dmg
echo "drag documentation into orange documentation mount drive"
echo "resize and convert with the Disk Copy utility"
open /Applications/Utilities/Disk\ Copy.app

