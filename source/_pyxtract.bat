@echo off
cd orange
alias python c:\python27\python
python ..\pyxtract\defvectors.py

rem I know this is stupid, but works under any shell
cd ..\orange
call _pyxtract.bat
cd ..\orangene
call _pyxtract.bat
cd ..\orangeom
call _pyxtract.bat
cd ..\orangol
call _pyxtract.bat
cd ..