!define NAME_PYTHON python-2.7.6.msi
!define NAME_PYTHONWIN pywin32-218.win32-py2.7.exe
!define NAME_MFC mfc71.dll

!macro modules
# To add additional modules call
# !insertmacro installmodule <human-readable name> <installer file (windows executable)> <file for checking whether the module is already installed>
	!insertmacro installmodule numpy numpy-1.8.1-win32-superpack-python2.7.exe $PythonDir\lib\site-packages\numpy\*.*
	!insertmacro installmodule scipy scipy-0.13.3-win32-superpack-python2.7.exe $PythonDir\lib\site-packages\scipy\*.*
	!insertmacro installmodule PIL PIL-1.1.7.win32-py2.7.exe $PythonDir\lib\site-packages\PIL\*.*
	!insertmacro installmodule PyQt PyQt-Py2.7-x86-gpl-4.9.6-1.exe $PythonDir\lib\site-packages\PyQt4\*.*
	!insertmacro installmodule PyQwt PyQwt5.2.0-Python2.7-PyQt4.9.6-NumPy1.8.1-1.exe $PythonDir\lib\site-packages\PyQt4\Qwt5\*.*
	!insertmacro installmodule setuptools setuptools-0.6c11.win32-py2.7.exe $PythonDir\lib\site-packages\setuptools\*.*
!macroend
