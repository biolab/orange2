Name "Orange"
Icon OrangeInstall.ico
UninstallIcon OrangeInstall.ico

!define PYFILENAME python-${NPYVER}.msi
!define PYWINFILENAME pywin32-212.win32-py${NPYVER}.exe

OutFile ${OUTFILENAME}

!include "LogicLib.nsh"

licensedata license.txt
licensetext "Acknowledgments and License Agreement"

AutoCloseWindow true
ShowInstDetails nevershow

Var PythonDir
Var AdminInstall
Var MissingModules

Page license
Page instfiles

!define SHELLFOLDERS \
  "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
 

Section Uninstall
	MessageBox MB_YESNO "Are you sure you want to remove Orange?$\r$\n$\r$\nThis won't remove any 3rd party software possibly installed with Orange, such as Python or Qt,$\r$\n$\r$\nbut make sure you have not left any of your files in Orange's directories!" /SD IDYES IDNO abort
	RmDir /R "$INSTDIR"
	${If} $AdminInstall = 0
	    SetShellVarContext all
	${Else}
	    SetShellVarContext current	   
	${Endif}
	RmDir /R "$SMPROGRAMS\Orange"

	ReadRegStr $0 HKCU "${SHELLFOLDERS}" AppData
	StrCmp $0 "" 0 +2
	  ReadRegStr $0 HKLM "${SHELLFOLDERS}" "Common AppData"
	StrCmp $0 "" +2 0
	  RmDir /R "$0\Orange"
	
	ReadRegStr $PythonDir HKLM Software\Python\PythonCore\${NPYVER}\InstallPath ""
	${If} $PythonDir != ""
		DeleteRegKey HKLM "SOFTWARE\Python\PythonCore\${NPYVER}\PythonPath\Orange"
		DeleteRegKey HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Orange"
	${Else}
		DeleteRegKey HKCU "SOFTWARE\Python\PythonCore\${NPYVER}\PythonPath\Orange"
		DeleteRegKey HKCU "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Orange"
	${Endif}
	
	Delete "$DESKTOP\Orange Canvas.lnk"

	DeleteRegKey HKEY_CLASSES_ROOT ".ows"
	DeleteRegKey HKEY_CLASSES_ROOT "OrangeCanvas"

	MessageBox MB_OK "Orange has been succesfully removed from your system.$\r$\nPython and other applications need to be removed separately.$\r$\n$\r$\nYou may now continue without rebooting your machine." /SD IDOK
  abort:
SectionEnd


!macro WarnMissingModule FILE MODULE
	${Unless} ${FileExists} ${FILE}
		${If} $MissingModules == ""
			StrCpy $MissingModules ${MODULE}
		${Else}
			StrCpy $MissingModules "$MissingModules, ${MODULE}"
		${EndIf}
	${EndUnless}
!macroend

!macro GetPythonDir
    ${If} $AdminInstall == 0
	    ReadRegStr $PythonDir HKCU Software\Python\PythonCore\${NPYVER}\InstallPath ""
		StrCmp $PythonDir "" 0 trim_backslash
		ReadRegStr $PythonDir HKLM Software\Python\PythonCore\${NPYVER}\InstallPath ""
		StrCmp $PythonDir "" return
		MessageBox MB_OK "Please ask the administrator to install Orange$\r$\n(this is because Python was installed by him, too)."
		Quit
	${Else}
	    ReadRegStr $PythonDir HKLM Software\Python\PythonCore\${NPYVER}\InstallPath ""
		StrCmp $PythonDir "" 0 trim_backslash
		ReadRegStr $PythonDir HKCU Software\Python\PythonCore\${NPYVER}\InstallPath ""
		StrCmp $PythonDir "" return
		StrCpy $AdminInstall 0
	${EndIf}

	trim_backslash:
	StrCpy $0 $PythonDir "" -1
    ${If} $0 == "\"
        StrLen $0 $PythonDir
        IntOp $0 $0 - 1
        StrCpy $PythonDir $PythonDir $0 0
    ${EndIf}

	return:
!macroend
		


!ifdef COMPLETE

!if ${PYVER} == 23
	!define MFC mfc42.dll
!else
	!define MFC mfc71.dll
!endif

Section ""
		StrCmp $PythonDir "" 0 have_python

		SetOutPath $DESKTOP
		!if ${PYVER} == 23
			askpython23:
				MessageBox MB_OKCANCEL "Orange installer will first launch installation of Python ${NPYVER}$\r$\nOrange installation will continue after you finish installing Python." /SD IDOK IDOK installpython23
					MessageBox MB_YESNO "Orange cannot run without Python.$\r$\nAbort the installation?" IDNO askpython23
						Quit
			installpython23:
				File ${PARTY}\Python-2.3.5.exe
				ExecWait "$DESKTOP\Python-2.3.5.exe"
				Delete "$DESKTOP\Python-2.3.5.exe"

		!else
		    StrCpy $0 ""
			askpython:
				MessageBox MB_YESNOCANCEL "Orange installer will first launch installation of Python ${NPYVER}.$\r$\nWould you like it to install automatically?$\r$\n(Press No for Custom installation of Python, Cancel to cancel installation of Orange." /SD IDYES IDYES installsilently IDNO installpython
					MessageBox MB_YESNO "Orange cannot run without Python.$\r$\nAbort the installation?" IDNO askpython
						Quit
			installsilently:
				StrCpy $0 "/Qb-"
			installpython:
				File ${PARTY}\${PYFILENAME}
				${If} $AdminInstall == 1
					ExecWait 'msiexec.exe /i "$DESKTOP\${PYFILENAME}" ADDLOCAL=Extensions,Documentation,TclTk ALLUSERS=1 $0' $0
				${Else}
					ExecWait 'msiexec.exe /i "$DESKTOP\${PYFILENAME}" ADDLOCAL=Extensions,Documentation,TclTk $0' $0
				${EndIf}
				Delete "$DESKTOP\${PYFILENAME}"
			!endif

		!insertMacro GetPythonDir
		StrCmp $PythonDir "" 0 have_python
			MessageBox MB_OK "Python installation failed.$\r$\nOrange installation cannot continue."
			Quit
	have_python:


		IfFileExists $PythonDir\lib\site-packages\PythonWin have_pythonwin
			MessageBox MB_YESNO "Do you want to install PythonWin (recommended)?$\r$\n(Orange installation will continue afterwards.)" /SD IDYES IDNO have_pythonwin
			IfFileExists "$SysDir\${MFC}" have_mfc
				SetOutPath $SysDir
				File ${PARTY}\${MFC}
			have_mfc:
			SetOutPath $DESKTOP
			File ${PARTY}\${PYWINFILENAME}
			ExecWait "$DESKTOP\${PYWINFILENAME}"
			Delete "$DESKTOP\${PYWINFILENAME}"
	have_pythonwin:

		!if ${QTVER} == 23
				SetOutPath $PythonDir\lib\site-packages
				IfFileExists $PythonDir\lib\site-packages\qt.py have_pyqt
					File /r ${PARTY}\pyqt\*.*
			have_pyqt:


				IfFileExists $PythonDir\lib\site-packages\qwt\*.* have_pyqwt
					File /r ${PARTY}\qwt
			have_pyqwt:


				IfFileExists $PythonDir\lib\site-packages\Numeric\*.* have_numeric
					File /r ${PARTY}\numeric
					File ${PARTY}\..\Numeric.pth
			have_numeric:


				IfFileExists $PythonDir\lib\site-packages\numpy\*.* have_numpy
					File /r ${PARTY}\numpy
			have_numpy:


				IfFileExists "$PythonDir\lib\site-packages\qt-mt230nc.dll" have_qt
				IfFileExists "$SysDir\qt-mt230nc.dll" have_qt
					File ${PARTY}\..\qt-mt230nc.dll
					SetOutPath $INSTDIR
					File ${PARTY}\..\QT-LICENSE.txt
			have_qt:
		!else
			MessageBox MB_OK "Installation will check for various needed libraries$\r$\nand launch their installers if needed."
			SetOutPath $DESKTOP
			
				IfFileExists $PythonDir\lib\site-packages\numpy-1.1.0-py2.5.egg-info have_numpy
				    File ${PARTY}\numpy-1.1.0-win32-superpack-python2.5.exe
					ExecWait $DESKTOP\numpy-1.1.0-win32-superpack-python2.5.exe
					Delete $DESKTOP\numpy-1.1.0-win32-superpack-python2.5.exe
					
			have_numpy:
				IfFileExists $PythonDir\lib\site-packages\PyQt4\*.* have_pyqt
				    File ${PARTY}\PyQt-Py2.5-gpl-4.4.2-1.exe
					ExecWait $DESKTOP\PyQt-Py2.5-gpl-4.4.2-1.exe
					Delete $DESKTOP\PyQt-Py2.5-gpl-4.4.2-1.exe
					
			have_pyqt:
				IfFileExists $PythonDir\lib\site-packages\PyQt4\Qwt5\*.* have_pyqwt
				    File ${PARTY}\PyQwt5.1.0-Python2.5-PyQt4.4.2-NumPy1.1.0-1.exe
					ExecWait $DESKTOP\PyQwt5.1.0-Python2.5-PyQt4.4.2-NumPy1.1.0-1.exe
					Delete $DESKTOP\PyQwt5.1.0-Python2.5-PyQt4.4.2-NumPy1.1.0-1.exe
			
			have_pyqwt:
		!endif
					
					
					
					
          
SectionEnd
!endif


Section ""
	ReadRegStr $0 HKCU "${SHELLFOLDERS}" AppData
	StrCmp $0 "" 0 +2
	  ReadRegStr $0 HKLM "${SHELLFOLDERS}" "Common AppData"
	StrCmp $0 "" not_installed_before 0

	IfFileExists "$0\Orange" 0 not_installed_before
		ask_remove_old:
		MessageBox MB_YESNOCANCEL "Another version of Orange has been found on the computer.$\r$\nDo you want to keep the existing settings for canvas and widgets?$\r$\n$\r$\nYou can usually safely answer 'Yes'; in case of problems, re-run this installation." /SD IDYES IDYES not_installed_before IDNO remove_old_settings
			MessageBox MB_YESNO "Abort the installation?" IDNO ask_remove_old
				Quit

		remove_old_settings:
		RmDir /R "$0\Orange"

	not_installed_before:

	StrCpy $INSTDIR  "$PythonDir\lib\site-packages\orange"
	SetOutPath $INSTDIR
	File /r /x .svn ${ORANGEDIR}\*

	!ifdef INCLUDEGENOMICS
	!endif

	!ifdef INCLUDETEXTMINING
		SetOutPath $PythonDir\lib\site-packages
		File E:\orange\download\snapshot\${PYVER}\lib\site-packages\orngText.pth
		File /r E:\orange\download\snapshot\${PYVER}\lib\site-packages\orngText

	!endif

	CreateDirectory "$SMPROGRAMS\Orange"
	CreateShortCut "$SMPROGRAMS\Orange\Orange for Beginners.lnk" "$INSTDIR\doc\ofb\default.htm"
	CreateShortCut "$SMPROGRAMS\Orange\Orange Modules Reference.lnk" "$INSTDIR\doc\modules\default.htm"
	CreateShortCut "$SMPROGRAMS\Orange\Orange Reference Guide.lnk" "$INSTDIR\doc\reference\default.htm"

	CreateShortCut "$SMPROGRAMS\Orange\Orange.lnk" "$INSTDIR\"
	CreateShortCut "$SMPROGRAMS\Orange\Uninstall Orange.lnk" "$INSTDIR\uninst.exe"

	SetOutPath $INSTDIR\OrangeCanvas
	CreateShortCut "$DESKTOP\Orange Canvas.lnk" "$INSTDIR\OrangeCanvas\orngCanvas.pyw" "" $INSTDIR\OrangeCanvas\icons\orange.ico 0
	CreateShortCut "$SMPROGRAMS\Orange\Orange Canvas.lnk" "$INSTDIR\OrangeCanvas\orngCanvas.pyw" "" $INSTDIR\OrangeCanvas\icons\orange.ico 0

	WriteRegStr SHELL_CONTEXT "SOFTWARE\Python\PythonCore\${NPYVER}\PythonPath\Orange" "" "$INSTDIR;$INSTDIR\OrangeWidgets;$INSTDIR\OrangeCanvas"
	WriteRegStr SHELL_CONTEXT "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange" "DisplayName" "Orange (remove only)"
	WriteRegStr SHELL_CONTEXT "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange" "UninstallString" '"$INSTDIR\uninst.exe"'

	WriteRegStr HKEY_CLASSES_ROOT ".ows" "" "OrangeCanvas"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\DefaultIcon" "" "$INSTDIR\OrangeCanvas\icons\OrangeOWS.ico"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\Shell\Open\Command\" "" '$PythonDir\python.exe $INSTDIR\orangeCanvas\orngCanvas.pyw "%1"'

	WriteUninstaller "$INSTDIR\uninst.exe"

	!ifdef INCLUDETEXTMINING
             ExecWait '"$PythonDir\python" -c $\"import orngRegistry; orngRegistry.addWidgetCategory(\$\"Text Mining\$\", \$\"$PythonDir\lib\site-packages\orngText\widgets\$\")$\"'
	!endif

SectionEnd	

Function .onInit
	StrCpy $AdminInstall 1

	UserInfo::GetAccountType
	Pop $1
	SetShellVarContext all
	${If} $1 != "Admin"
		SetShellVarContext current
		StrCpy $AdminInstall 0
	${Else}
		SetShellVarContext all
		StrCpy $AdminInstall 1
	${EndIf}

	!insertMacro GetPythonDir


	!ifndef COMPLETE
		StrCmp $PythonDir "" 0 have_python
			MessageBox MB_OK "Please install Python first (www.python.org)$\r$\nor download Orange distribution that includes Python."
			Quit
		have_python:

	!if ${QTVER} == 23
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\qt.py" "PyQt"
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\qwt\*.*" "PyQwt"
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\Numeric\*.*" "Numeric"
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\numpy\*.*" "numpy"
		IfFileExists "$SYSDIR\qt-mt230nc.dll" have_qt
			!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\qt-mt230nc.dll" "Qt"
        have_qt:
	!else
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\PyQt4\*.*" "PyQt"
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\numpy-1.1.0-py2.5.egg-info" "numpy"
		!insertMacro WarnMissingModule "$PythonDir\lib\site-packages\PyQt4\Qwt5\*.*" "PyQwt"
	!endif

		StrCmp $MissingModules "" continueinst
		MessageBox MB_YESNO "Missing module(s): $MissingModules$\r$\n$\r$\nWithout these modules you can still scripts in Orange, but Orange Canvas will not work without them.$\r$\nYou can download and install them later or obtain the Orange installation that includes them.$\r$\n$\r$\nContinue with installation?" /SD IDYES IDYES continueinst
		Quit
		continueinst:
	!endif
FunctionEnd


Function .onInstSuccess
	MessageBox MB_OK "Orange has been successfully installed." /SD IDOK
FunctionEnd
