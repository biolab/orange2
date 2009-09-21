!define NAME_PYTHON python-2.5.msi
!define NAME_PYTHONWIN pywin32-212.win32-py2.5.exe
!define NAME_MFC mfc71.dll

!macro modules
# To add additional modules call
# !insertmacro installmodule <human-readable name> <installer file (windows executable)> <file for checking whether the module is already installed>
	!installmodule numpy numpy-1.1.0-win32-superpack-python2.5.exe $PythonDir\lib\site-packages\numpy-1.1.0-py2.5.egg-info
	!installmodule PIL PIL-1.1.6.win32-py2.5.exe $PythonDir\lib\site-packages\PIL\*.*
	!installmodule PyQt PyQt-Py2.5-gpl-4.4.2-1.exe $PythonDir\lib\site-packages\PyQt4\*.*
	!installmodule PyQwt PyQwt5.1.0-Python2.5-PyQt4.4.2-NumPy1.1.0-1.exe $PythonDir\lib\site-packages\PyQt4\Qwt5\*.*
!macroend
    