pushd
cd ..\..
svn update doc
cd doc\widgets\catalog
move default-web.htm default.htm
python ..\..\..\scripts\doc\widgetCatalogs.py html widgetRegistry.xml . >> default.htm
popd
