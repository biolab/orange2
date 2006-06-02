"C:\Program Files\Microsoft Visual Studio\VC98\Bin\vcvars32.bat" 
set PYTHON24=c:\python24

cd Orange
nmake /f  Orange.mak CFG="Orange - Win32 Python 24"
cd ..

cd Corn
nmake /f  Corn.mak CFG="Corn - Win32 Python 24"
cd ..

cd Orangeom
nmake /f  Orangeom.mak CFG="orangeom - Win32 Python 24"
cd ..

cd Orangene
nmake /f  Orangene.mak CFG="orangene - Win32 Python 24"
cd ..

cd Statc
nmake /f  Statc.mak CFG="Statc - Win32 Python 24"
cd ..
