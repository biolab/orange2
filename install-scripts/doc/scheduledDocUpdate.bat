cd ..\..
"c:\program files\wincvs 1.3\cvsnt\cvs.exe" -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS update -d doc
cd doc\scripts
call makeLocalCatalog.bat
