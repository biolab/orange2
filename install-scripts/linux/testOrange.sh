TAG=HEAD

rm -f orange.pth
find `pwd`/orange -name '*' -type d | grep -v CVS | grep -v source > orange.pth

rm -Rf regressionTests
cvs -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS checkout -r $TAG -f regressionTests

# fix the path for the regression test from ../doc to ../orange/doc
cat regressionTests/xtest.py | sed s/"os.chdir(\"..\/doc\")"/"os.chdir(\"..\/orange\/doc\")"/ > regressionTests/new.py
mv -f regressionTests/new.py regressionTests/xtest.py

cd regressionTests
if ! python ./xtest.py test; then
	cd ..
	exit 1
else
	cd ..
	exit 0
fi

