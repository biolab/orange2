call "C:\Program Files\Microsoft Visual Studio\VC98\Bin\vcvars32.bat" 
set PYTHON24=c:\python24

cd Orange
del *.dep
..\..\pyxtract\fixmak.py Orange.mak
nmake /f  temp.mak CFG="Orange - Win32 Python 24"
del temp.mak
cd ..

cd Corn
del *.dep
..\..\pyxtract\fixmak.py Corn.mak
nmake /f  temp.mak CFG="Corn - Win32 Python 24"
del temp.mak
cd ..

cd Orangeom
del *.dep
..\..\pyxtract\fixmak.py Orangeom
nmake /f  temp.mak CFG="orangeom - Win32 Python 24"
del temp.mak
cd ..

cd Orangene
del *.dep
..\..\pyxtract\fixmak.py Orangene.mak.mak
nmake /f  temp.mak CFG="orangene - Win32 Python 24"
del temp.mak
cd ..

cd Statc
del *.dep
..\..\pyxtract\fixmak.py Statc.mak
nmake /f  temp.mak CFG="Statc - Win32 Python 24"
del temp.mak
cd ..
