!define NAME_PYTHON python-2.6.2.msi
!define NAME_PYTHONWIN pywin32-214.win32-py2.6.exe
!define NAME_MFC mfc71.dll

!macro modules
# To add additional modules call
# !insertmacro installmodule <human-readable name> <installer file (windows executable)> <file for checking whether the module is already installed>
	!insertmacro installmodule numpy numpy-1.3.0-win32-superpack-python2.6.exe $PythonDir\lib\site-packages\numpy-1.3.0-py2.6.egg-info
	!insertmacro installmodule PIL PIL-1.1.6.win32-py2.6.exe $PythonDir\lib\site-packages\PIL\*.*
	!insertmacro installmodule PyQt PyQt-Py2.6-gpl-4.5.4-1.exe $PythonDir\lib\site-packages\PyQt4\*.*
	!insertmacro installmodule PyQwt PyQwt5.2.0-Python2.6-PyQt4.5.4-NumPy1.3.0-1.exe $PythonDir\lib\site-packages\PyQt4\Qwt5\*.*
!macroend
