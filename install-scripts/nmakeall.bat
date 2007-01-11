pushd

set PYTHON=c:\python%1

cdd c:\
del /efkqsxyz \compileTemp
cd compileTemp

cvs -d :sspi:janez@estelle.fri.uni-lj.si:/CVS checkout -A -- source

cd source
cmd /c _pyxtract.bat


call "C:\Program Files\Microsoft Visual Studio\VC98\Bin\vcvars32.bat" 

for %lll in (include Orange Corn orangeom orangene Statc) (
  cd %lll %+
  ..\pyxtract\fixmak.py %lll.mak %+
  if not exist obj mkdir obj %+
  nmake /f  temp.mak CFG="%lll - Win32 Release" %+
  cd ..
)

cd \
if not exist compiled mkdir compiled
if not exist compiled\%1 mkdir compiled\%1
copy compileTemp\*.pyd compiled\%1

popd
