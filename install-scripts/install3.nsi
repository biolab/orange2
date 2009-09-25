Name "Orange"
Icon OrangeInstall.ico
UninstallIcon OrangeInstall.ico
licensedata license.txt
licensetext "Acknowledgments and License Agreement"

OutFile ${OUTFILENAME}

!include "LogicLib.nsh"

!ifdef COMPLETE
	!macro installmodule modulename installfile checkfile
		${Unless} ${FileExists} ${checkfile}
			File ${PARTY}\${installfile}
			ExecWait $DESKTOP\${installfile}
			Delete $DESKTOP\${installfile}
		${EndUnless}
	!macroend
!else
	Var MissingModules
	!macro installmodule modulename installfile checkfile
		${Unless} ${FileExists} ${checkfile}
		${AndUnless} modulename == ""
			${If} $MissingModules == ""
				StrCpy $MissingModules ${modulename}
			${Else}
				StrCpy $MissingModules "$MissingModules, ${modulename}"
			${EndIf}
		${EndUnless}
	!macroend
!endif

!include "${PARTY}\names.inc"

AutoCloseWindow true
ShowInstDetails nevershow

Var PythonDir
Var AdminInstall

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

Section ""
		StrCmp $PythonDir "" 0 have_python

		SetOutPath $DESKTOP
		StrCpy $0 ""
		askpython:
			MessageBox MB_YESNOCANCEL "Orange installer will first launch installation of Python ${NPYVER}.$\r$\nWould you like it to install automatically?$\r$\n(Press No for Custom installation of Python, Cancel to cancel installation of Orange." /SD IDYES IDYES installsilently IDNO installpython
				MessageBox MB_YESNO "Orange cannot run without Python.$\r$\nAbort the installation?" IDNO askpython
					Quit
		installsilently:
			StrCpy $0 "/Qb-"
		installpython:
			File ${PARTY}\${NAME_PYTHON}
			${If} $AdminInstall == 1
				ExecWait 'msiexec.exe /i "$DESKTOP\${NAME_PYTHON}" ALLUSERS=1 $0' $0
			${Else}
				ExecWait 'msiexec.exe /i "$DESKTOP\${NAME_PYTHON}" $0' $0
			${EndIf}
			Delete "$DESKTOP\${NAME_PYTHON}"
		
			!insertMacro GetPythonDir
			StrCmp $PythonDir "" 0 have_python
				MessageBox MB_OK "Python installation failed.$\r$\nOrange installation cannot continue."
				Quit

		have_python:

		IfFileExists $PythonDir\lib\site-packages\PythonWin have_pythonwin
			MessageBox MB_YESNO "Do you want to install PythonWin (recommended)?$\r$\n(Orange installation will continue afterwards.)" /SD IDYES IDNO have_pythonwin
			IfFileExists "$SysDir\${NAME_MFC}" have_mfc
				SetOutPath $SysDir
				File ${PARTY}\${NAME_MFC}
			have_mfc:
			SetOutPath $DESKTOP
			File ${PARTY}\${NAME_PYTHONWIN}
			ExecWait "$DESKTOP\${NAME_PYTHONWIN}"
			Delete "$DESKTOP\${NAME_PYTHONWIN}"
			
		have_pythonwin:

		MessageBox MB_OK "Installation will check for various needed libraries$\r$\nand launch their installers if needed."
		SetOutPath $DESKTOP
		!insertMacro modules
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

		!insertMacro modules
		StrCmp $MissingModules "" continueinst
			MessageBox MB_YESNO "Missing module(s): $MissingModules$\r$\n$\r$\nWithout these modules you can still scripts in Orange, but Orange Canvas will not work without them.$\r$\nYou can download and install them later or obtain the Orange installation that includes them.$\r$\n$\r$\nContinue with installation?" /SD IDYES IDYES continueinst
			Quit

		continueinst:
	!endif
FunctionEnd


Function .onInstSuccess
	MessageBox MB_OK "Orange has been successfully installed." /SD IDOK
FunctionEnd
