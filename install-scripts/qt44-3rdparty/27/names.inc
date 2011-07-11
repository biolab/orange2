!define NAME_PYTHON python-2.7.1.msi
!define NAME_PYTHONWIN pywin32-214.win32-py2.7.exe
!define NAME_MFC mfc71.dll

!macro modules
# To add additional modules call
# !insertmacro installmodule <human-readable name> <installer file (windows executable)> <file for checking whether the module is already installed>
	!insertmacro installmodule numpy numpy-1.5.1-win32-superpack-python2.7.exe $PythonDir\lib\site-packages\numpy-1.5.1-py2.7.egg-info
	!insertmacro installmodule PIL PIL-1.1.7.win32-py2.7.exe $PythonDir\lib\site-packages\PIL\*.*
	!insertmacro installmodule PyQt PyQt-Py2.7-gpl-4.8.1-1.exe $PythonDir\lib\site-packages\PyQt4\*.*
	!insertmacro installmodule PyQwt PyQt4.Qwt5-5.2.1.win32-py27.exe $PythonDir\lib\site-packages\PyQt4\Qwt5\*.*
	!insertmacro installmodule networkx networkx-1.5.win32.exe $PythonDir\lib\site-packages\networkx\*.*
!macroend
