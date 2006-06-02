call "C:\Program Files\Microsoft Visual Studio\VC98\Bin\vcvars32.bat" 
set PYTHON24=c:\python24

cd Orange
..\pyxtract\fixmak.py Orange.mak
nmake /f  temp.mak CFG="Orange - Win32 Python 24"
nmake /f  temp.mak CFG="Orange - Win32 Release"
del temp.mak
cd ..

cd Corn
..\pyxtract\fixmak.py Corn.mak
nmake /f  temp.mak CFG="Corn - Win32 Python 24"
nmake /f  temp.mak CFG="Corn - Win32 Release"
del temp.mak
cd ..

cd Orangeom
..\pyxtract\fixmak.py Orangeom.mak
nmake /f  temp.mak CFG="orangeom - Win32 Python 24"
nmake /f  temp.mak CFG="orangeom - Win32 Release"
del temp.mak
cd ..

cd Orangene
..\pyxtract\fixmak.py Orangene.mak
nmake /f  temp.mak CFG="orangene - Win32 Python 24"
nmake /f  temp.mak CFG="orangene - Win32 Release"
del temp.mak
cd ..

cd Statc
..\pyxtract\fixmak.py Statc.mak
nmake /f  temp.mak CFG="Statc - Win32 Python 24"
nmake /f  temp.mak CFG="orangene - Win32 Release"
del temp.mak
cd ..
