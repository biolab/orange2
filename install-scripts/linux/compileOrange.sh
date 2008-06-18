## export the latest version
TAG=stable

gcc -v
echo
echo "checking out sources using tag: $TAG"
if [ "$1" == "clean" ]; then
	## force a clean checkout
	## compiling this might take some time
	echo "force complete compile"
	rm -Rf source 
fi
echo

rm -Rf orange
rm -Rf orange/source
cvs -q -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS checkout -r $TAG -f orange
mkdir orange
mkdir orange/source
cd orange
cvs -q -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS checkout -r $TAG -f -d source source
cd ..

START_WD=`pwd`
cd orange
rm -f orange.so statc.so corn.so orangene.so orangeom.so
cd source 
if ! make; then 
	echo -e "\n\nERROR compiling"
	exit 1
else
	echo -e "\n\nOrange compiled successfully"
fi
cd $START_WD
