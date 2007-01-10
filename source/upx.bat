if exist ..\..\%1.pyd del ..\..\%1.pyd
"c:\program files\upx" obj\release\%1.pyd -o ..\..\%1.pyd
copy obj\Release\%1.lib ..\..\lib\%1.lib