call _pyxtract.bat

set PYTHON=c:\python24
call nmakeall.bat
mkdir 24
copy *.pyd 24

set PYTHON=c:\python23
call nmakeall.bat
mkdir 23
copy *.pyd 23

set PYTHON=c:\python25
call nmakeall.bat
mkdir 25
copy *.pyd 25

cd source