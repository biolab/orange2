TAG=HEAD

cd /home/orange/daily/test_install/orange
cvs -Q -d :pserver:cvso@estelle.fri.uni-lj.si:/cvs checkout -d . -r $TAG -f regressionTests

# fix the path for the regression test from ../doc to ../orange/doc
cat xtest.py | sed s/"\.\.\/doc"/"\.\.\/doc\/orange"/ > new.py
mv -f new.py xtest.py

if ! python xtest.py test; then
	exit 1
else
	exit 0
fi

