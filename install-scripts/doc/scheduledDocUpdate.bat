cd ..\..
"c:\program files\wincvs 1.3\cvsnt\cvs.exe" -d :pserver:cvso@estelle.fri.uni-lj.si:/CVS update -d doc
cd doc\widgets\catalog
move default-web.htm default.htm
python ..\..\..\scripts\doc\widgetCatalogs.py html widgetRegistry.xml . >> default.htm
