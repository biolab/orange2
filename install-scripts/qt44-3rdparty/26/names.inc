!define PYTHONNAME python-2.6.2.msi
!define PYTHONWINNAME pywin32-214.win32-py2.6.exe
!define MFCNAME mfc71.dll

!macro installmodules
	!installmodule "" mfc71.dll "$SysDir\${MFC}"
	!installmodule numpy numpy-1.3.0-win32-superpack-python2.6.exe $PythonDir\lib\site-packages\numpy-1.3.0-py2.6.egg-info
	!installmodule PIL PIL-1.1.6.win32-py2.6.exe $PythonDir\lib\site-packages\PIL\*.*
	!installmodule PyQt PyQt-Py2.6-gpl-4.5.4-1.exe $PythonDir\lib\site-packages\PyQt4\*.*
	!installmodule PyQwt PyQwt5.2.0-Python2.6-PyQt4.5.4-NumPy1.3.0-1.exe $PythonDir\lib\site-packages\PyQt4\Qwt5\*.*
	!installmodule 
!macroend
