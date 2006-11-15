cd ..\..\doc\widgets\catalog
copy default-chm.htm default.htm
python ..\..\..\scripts\doc\widgetCatalogs.py htmlverb ..\..\..\orangeCanvas\widgetRegistry.xml . >> default.htm
