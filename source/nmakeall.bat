call "C:\Program Files\Microsoft Visual Studio\VC98\Bin\vcvars32.bat" 

cd include
..\pyxtract\fixmak.py include.mak
mkdir obj
nmake /f  temp.mak CFG="include - Win32 Release"
rem del temp.mak
cd ..

cd Orange
..\pyxtract\fixmak.py Orange.mak
mkdir obj
nmake /f  temp.mak CFG="Orange - Win32 Release"
rem del temp.mak
cd ..

exit

cd Corn
..\pyxtract\fixmak.py Corn.mak
nmake /f  temp.mak CFG="Corn - Win32 Release"
rem del temp.mak
cd ..

cd Orangeom
..\pyxtract\fixmak.py Orangeom.mak
nmake /f  temp.mak CFG="orangeom - Win32 Release"
rem del temp.mak
cd ..

cd Orangene
..\pyxtract\fixmak.py Orangene.mak
nmake /f  temp.mak CFG="orangene - Win32 Release"
rem del temp.mak
cd ..

cd Statc
..\pyxtract\fixmak.py Statc.mak
nmake /f  temp.mak CFG="orangene - Win32 Release"
rem del temp.mak
cd ..
