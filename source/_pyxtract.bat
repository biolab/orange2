@echo off
cd orange
python ..\pyxtract\defvectors.py
cd ..
for %%d in (orange orangene orangeom) do pushd %%d & call _pyxtract.bat & popd
