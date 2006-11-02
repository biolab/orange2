cd ..\..\doc\widgets\catalog
copy default-chm.htm default.htm
python ..\..\..\scripts\doc\widgetCatalogs.py html ..\..\..\orangeCanvas\widgetRegistry.xml . >> default.htm
