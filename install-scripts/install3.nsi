Name "Orange"
Icon OrangeInstall.ico
UninstallIcon OrangeInstall.ico

!ifndef ORANGEDIR
	!define ORANGEDIR orange
!endif

!ifdef COMPLETE
  !ifndef OUTFILENAME
		OutFile "Orange-complete.exe"
	!endif
	!define INCLUDEPYTHON
;	!define INCLUDEPYTHONWIN
	!define INCLUDEPYQT
	!define INCLUDEPYQWT
	!define INCLUDENUMERIC
	!define INCLUDEQT
	!define INCLUDESCRIPTDOC
	!define INCLUDEDATASETS
!else
!ifdef STANDARD		; orange (*.py *.pyd) and doc only
  !ifndef OUTFILENAME
		OutFile "Orange-standard.exe"
	!endif
	!define INCLUDESCRIPTDOC
!endif
!endif

!ifdef OUTFILENAME
OutFile ${OUTFILENAME}
!else
OutFile "orange-temp.exe"
!endif


!include "LogicLib.nsh"

licensedata license.txt
licensetext "Acknowledgments and License Agreement"

; InstallDirRegKey HKEY_LOCAL_MACHINE "SOFTWARE\Python\PythonCore\2.3\PythonPath\Orange" ""

AutoCloseWindow true
ShowInstDetails nevershow
SilentUninstall silent

Var PythonDir
Var PythonOnDesktop

Page license
Page directory
Page components
Page instfiles

!ifdef INCLUDEPYQT | INCLUDEPYQWT | INCLUDENUMERIC

	ComponentText "Components" "Select components to install" "(The Python stuff that you already have is hidden)"

	Subsection /e "!" SSPYTHON

	!ifdef INCLUDEPYQT
		Section "PyQt" SECPYQT
			SetOutPath $PythonDir\lib\site-packages
			File /r pyqt\*.*
		SectionEnd
	!endif

	!ifdef INCLUDEPYQWT
		Section "PyQwt" SECPYQWT
			SetOutPath $PythonDir\lib\site-packages
			File /r qwt
		SectionEnd
	!endif

	!ifdef INCLUDENUMERIC
		Section "Numeric Python" SECNUMERIC
			SetOutPath $PythonDir\lib\site-packages
			File /r numeric
			File various\Numeric.pth
		SectionEnd
	!endif

	SubsectionEnd
	
!else ; no Python modules included with this installation
	ComponentText "Components" "" "Select components to install"
!endif


!ifdef INCLUDEQT
	Section "Qt 2.2 non-commercial" SECQT
		SetOutPath $SYSDIR
		File various\qt-mt230nc.dll

		SetOutPath $INSTDIR
		File various\QT-LICENSE.txt
	SectionEnd
!endif


Section "Orange Modules"
    !include 
	SetOutPath $INSTDIR
	File ${ORANGEDIR}\*.py
	File ${ORANGEDIR}\*.pyd
	SetOutPath $INSTDIR\icons
	File Orange.ico
	
	SetOutPath $INSTDIR\OrangeWidgets
	File /r ${ORANGEDIR}\OrangeWidgets\*.py
	File /r ${ORANGEDIR}\OrangeWidgets\*.png
	SetOutPath $INSTDIR\OrangeCanvas
	File /r ${ORANGEDIR}\OrangeCanvas\*.py
	File /r ${ORANGEDIR}\OrangeCanvas\*.png

	SetOutPath $INSTDIR\OrangeCanvas\icons
	File OrangeOWS.ico
SectionEnd


!ifdef INCLUDEGENOMICS
Section "Genomic Data"
	SetOutPath $INSTDIR\doc
	File "various\Orange Genomics.pdf"

	SetOutPath $INSTDIR
	CreateDirectory "$SMPROGRAMS\Orange"
	CreateShortCut "$SMPROGRAMS\Orange\Orange Widgets For Functional Genomics.lnk" "$INSTDIR\doc\Orange Genomics.pdf"

	SetOutPath $INSTDIR\OrangeWidgets\Genomics
	File /r ${ORANGEDIR}\..\Genomics\GO
	File /r ${ORANGEDIR}\..\Genomics\Annotation
	File /r "${ORANGEDIR}\..\Genomics\Genome Map"
	SetOutPath "$INSTDIR\OrangeCanvas"
	File various\bi-visprog\*.tab
	File various\bi-visprog\*.ows
SectionEnd
!endif
	

!ifdef INCLUDESCRIPTDOC | INCLUDEDATASETS
	Subsection /e "Documentation"

	!ifdef INCLUDESCRIPTDOC
		Section "Scripting Documentation"
			SetOutPath $INSTDIR\doc
			File "various\Orange White Paper.pdf"
			File "various\Orange Widgets White Paper.pdf"

			!cd ${ORANGEDIR}\doc
			SetOutPath $INSTDIR\doc
			File /r ofb
			File /r modules
			File /r reference
			File style.css
			SetOutPath $INSTDIR
                        CreateDirectory "$SMPROGRAMS\Orange"
			CreateShortCut "$SMPROGRAMS\Orange\Orange White Paper.lnk" "$INSTDIR\doc\Orange White Paper.pdf"
			CreateShortCut "$SMPROGRAMS\Orange\Orange Widgets White Paper.lnk" "$INSTDIR\doc\Orange Widgets White Paper.pdf"
			CreateShortCut "$SMPROGRAMS\Orange\Orange for Beginners.lnk" "$INSTDIR\doc\ofb\default.htm"
			CreateShortCut "$SMPROGRAMS\Orange\Orange Modules Reference.lnk" "$INSTDIR\doc\modules\default.htm"
			CreateShortCut "$SMPROGRAMS\Orange\Orange Reference Guide.lnk" "$INSTDIR\doc\reference\default.htm"
			!cd ${CWD}
			!echo "CHANGING TO ${CWD}"
		SectionEnd
	!endif
  
	!ifdef INCLUDEDATASETS
		Section "Datasets"
			SetOutPath $INSTDIR\doc\datasets
			File /r ${ORANGEDIR}\doc\datasets\*
			SectionEnd
	!endif

	SubsectionEnd
!endif ;  | INCLUDESCRIPTDOC INCLUDEDATASETS

!ifdef INCLUDESOURCE
	Section "Orange Source"
		SetOutPath $INSTDIR
		File /r ${ORANGEDIR}\source
	SectionEnd
!endif

Section ""
	SetOutPath $INSTDIR
	
	Delete $INSTDIR\OrangeWidgets\OWLin_Results.py
	Delete $INSTDIR\OrangeWidgets\Visualize\OWLinViz.py
	Delete $INSTDIR\OrangeWidgets\Data\OWSelectAttributes.py
	Delete $INSTDIR\OrangeWidgets\Classify\OWLearner.py
	Delete $INSTDIR\OrangeWidgets\Classify\OWCalibratedClassifier.py
	Delete $INSTDIR\OrangeWidgets\Data\OWExampleBuilder.py
	Delete $INSTDIR\OrangeWidgets\Data\OWSubsetGenerator.py
	Delete $INSTDIR\OrangeWidgets\Other\OWITree.py
	
	CreateDirectory "$SMPROGRAMS\Orange"
	CreateShortCut "$SMPROGRAMS\Orange\Orange.lnk" "$INSTDIR\"
	CreateShortCut "$SMPROGRAMS\Orange\Uninstall Orange.lnk" "$INSTDIR\uninst.exe"
	SetOutPath $INSTDIR\OrangeCanvas
	CreateShortCut "$DESKTOP\Orange Canvas.lnk" "$INSTDIR\OrangeCanvas\orngCanvas.py" "" $INSTDIR\icons\Orange.ico 0
	CreateShortCut "$SMPROGRAMS\Orange\Orange Canvas.lnk" "$INSTDIR\OrangeCanvas\orngCanvas.py" "" $INSTDIR\icons\Orange.ico 0

	WriteRegStr HKEY_LOCAL_MACHINE "SOFTWARE\Python\PythonCore\2.3\PythonPath\Orange" "" "$INSTDIR;$INSTDIR\OrangeWidgets;$INSTDIR\OrangeCanvas"
	WriteRegStr HKEY_LOCAL_MACHINE "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange" "DisplayName" "Orange (remove only)"
	WriteRegStr HKEY_LOCAL_MACHINE "Software\Microsoft\Windows\CurrentVersion\Uninstall\Orange" "UninstallString" '"$INSTDIR\uninst.exe"'
	
	;ows icon and association, schema-click launch
	WriteRegStr HKEY_CLASSES_ROOT ".ows" "" "OrangeCanvas"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\DefaultIcon" "" "$INSTDIR\OrangeCanvas\icons\OrangeOWS.ico"
	WriteRegStr HKEY_CLASSES_ROOT "OrangeCanvas\Shell\Open\Command\" "" '$PythonDir\python.exe $INSTDIR\orangeCanvas\orngCanvas.py "%1"'

	WriteUninstaller "$INSTDIR\uninst.exe"
SectionEnd  

Section Uninstall
	MessageBox MB_YESNO "Are you sure you want to remove Orange?$\r$\n(This won't remove any 3rd party software possibly installed with Orange, such as Python or Qt)?$\r$\n$\r$\nMake sure you have not left any of your files in Orange's directories!" IDNO abort
	RmDir /R "$INSTDIR"
	RmDir /R "$SMPROGRAMS\Orange"
	DeleteRegKey HKEY_LOCAL_MACHINE "SOFTWARE\Python\PythonCore\2.3\PythonPath\Orange"
	DeleteRegKey HKEY_LOCAL_MACHINE "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Orange"
	Delete "$DESKTOP\Orange Canvas.lnk"

	; remove owc icon and file type associations
	DeleteRegKey HKEY_CLASSES_ROOT ".ows"
	DeleteRegKey HKEY_CLASSES_ROOT "OrangeCanvas"

	MessageBox MB_OK "Orange has been succesfully removed from your system.$\r$\nPython and other applications need to be removed separately.$\r$\n$\r$\nYou may now continue without rebooting your machine."
  abort:
SectionEnd


!macro HideSection SECTION
	SectionGetFlags ${SECTION} $0
	IntOp $0 $0 & 0xFFFFFFFE ; disable
	IntOp $0 $0 | 0x00000010 ; readonly
	SectionSetFlags ${SECTION} $0
	SectionSetText ${SECTION} ""
!macroend


!ifdef INCLUDEPYQT | INCLUDEPYQWT | INCLUDENUMERIC
	!macro DisEnSection SECTION
			!insertMacro HideSection ${SECTION}
		${Else}
			SectionSetText ${SSPYTHON} "Python Modules"
	!macroend
!endif

Function .onGUIInit
	StrCpy $PythonOnDesktop 0

	ReadRegStr $PythonDir HKLM Software\Python\PythonCore\2.3\InstallPath ""

	${If} $PythonDir S== ""
	
		!ifdef INCLUDEPYTHON
		  askpython:
			MessageBox MB_OKCANCEL "Orange installer will first launch installation of Python (ver 2.3.2-1)$\r$\nOrange installation will continue after you finish installing Python." IDOK installpython
			MessageBox MB_YESNO "Orange cannot run without Python.$\r$\nAbort the installation?" IDNO askpython
			Quit
		
		  installpython:
			SetOutPath $DESKTOP
			File various\Python-2.3.2-1.exe
			StrCpy $PythonOnDesktop 1
			ExecWait "$DESKTOP\Python-2.3.2-1.exe"

			ReadRegStr $PythonDir HKLM Software\Python\PythonCore\2.3\InstallPath ""
			${If} $PythonDir S== ""
				MessageBox MB_OK "Python installation failed.$\r$\nOrange installation cannot continue."
				Quit
			${EndIf}
			
		!else
			MessageBox MB_OK "Cannot find Python 2.3.$\r$\nDownload it from www.python.org and install, or$\r$\nget an Orange distribution that includes Python"
			Quit
		!endif

		; let the user select the modules
		!ifdef INCLUDEPYTHONWIN | INCLUDEPYQT | INCLUDEPYQWT | INCLUDENUMERIC
		SectionSetText ${SSPYTHON} "Python Modules" 
		!endif
	${Else}
		; we have Python already - let's check the modules

		!ifdef INCLUDEPYQT
			${If} ${FileExists} $PythonDir\lib\site-packages\qt.py
				!insertMacro DisEnSection ${SECPYQT}
			${EndIf}
		!endif

		!ifdef INCLUDEPYQWT
			${If} ${FileExists} $PythonDir\lib\site-packages\qwt\*.*
				!insertMacro DisEnSection ${SECPYQWT}
			${EndIf}
		!endif

		!ifdef INCLUDENUMERIC
			${If} ${FileExists} $PythonDir\lib\site-packages\Numeric\*.*
				!insertMacro DisEnSection ${SECNUMERIC}
			${EndIf}
		!endif
		
	${EndIf}


	!ifdef INCLUDEPYTHONWIN
		ReadRegStr $8 HKLM Software\Python\PythonCore\2.3\PythonPath\PythonWin ""
		${If} $8 S== ""
			MessageBox MB_YESNO "Do you want to install PythonWin?$\r$\n(recommended if you plan programming scripts)" IDNO dontinstallpythonwin
			SetOutPath $DESKTOP
			File various\win32all-163.exe
			StrCpy $PythonOnDesktop 1
			ExecWait "$DESKTOP\win32all-163.exe"

			ReadRegStr $8 HKLM Software\Python\PythonCore\2.3\PythonPath\PythonWin ""
			${If} $8 S== ""
				MessageBox MB_OK "PythonWin installation failed.$\r$\nOrange installation will now resume."
			${EndIf}

		${EndIf}
	    dontinstallpythonwin:
	!endif

	
	!ifdef INCLUDEQT
		${If} ${FileExists} "$SYSDIR\qt-mt230nc.dll"
			!insertMacro HideSection ${SECQT}
		${EndIf}
	!endif
	
	StrCpy $INSTDIR $PythonDir\lib\site-packages\orange
FunctionEnd


Function .onInstSuccess
	${If} $PythonOnDesktop == 1
		MessageBox MB_OK "Orange has been successfully installed.$\r$\n$\r$\nPython installation files have been put on the desktop$\r$\nin case you may want to store them."
	${Else}
		MessageBox MB_OK "Orange has been successfully installed."
	${EndIf}
FunctionEnd