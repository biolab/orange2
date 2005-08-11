TAG=HEAD

#rm -f orange.pth
#find `pwd`/orange -name '*' -type d | grep -v CVS | grep -v source > orange.pth

cvs -q -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS checkout -d /home/orange/test_download/install/orange -r $TAG -f regressionTests

# fix the path for the regression test from ../doc to ../orange/doc
cat /home/orange/test_download/install/orange/xtest.py | sed s/"os.chdir(\"..\/doc/\")"/"os.chdir(\"..\/doc\/orange\")"/ > /home/orange/test_download/install/orange/new.py
mv -f /home/orange/test_download/install/orange/new.py /home/orange/test_download/install/orange/xtest.py

cd /home/orange/test_download/install/orange/
if ! python ./xtest.py test; then
	cd ..
	exit 1
else
	cd ..
	exit 0
fi

