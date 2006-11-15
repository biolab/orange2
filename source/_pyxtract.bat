@echo off
cd orange
python ..\pyxtract\defvectors.py
cd ..
for %%d in (orange orangene orangeom) do cd %%d & call _pyxtract.bat & cd ..
pause
