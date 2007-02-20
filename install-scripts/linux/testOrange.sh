TAG=HEAD

cd /home/orange/daily/test_install/orange
cvs -Q -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -r $TAG -f -d regressionTests regressionTests
mv regressionTests/* .

# fix the path for the regression test from ../doc to ../orange/doc
cat xtest.py | sed s/"\.\.\/doc"/"\.\.\/doc\/orange"/ > new.py
mv -f new.py xtest.py

LD_LIBRARY_PATH=/home/orange/daily/test_install/orange:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

# set time limit for regression tests
ulimit -t 1800
if ! python xtest.py test; then
	exit 1
else
	exit 0
fi

# commit testresults.xml
# cvs commit -m "" testresults.xml

