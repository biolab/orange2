# Microsoft Developer Studio Generated NMAKE File, Based on orangeom.dsp
!IF "$(CFG)" == ""
CFG=orangeom - Win32 Debug
!MESSAGE No configuration specified. Defaulting to orangeom - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "orangeom - Win32 Release" && "$(CFG)" != "orangeom - Win32 Debug" && "$(CFG)" != "orangeom - Win32 Release_Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "orangeom.mak" CFG="orangeom - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "orangeom - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "orangeom - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "orangeom - Win32 Release_Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "orangeom - Win32 Release"

OUTDIR=.\obj/Release
INTDIR=.\obj/Release
# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

ALL : "$(OUTDIR)\orangeom.pyd"


CLEAN :
	-@erase "$(INTDIR)\datafile.obj"
	-@erase "$(INTDIR)\fileio.obj"
	-@erase "$(INTDIR)\graphDrawing.obj"
	-@erase "$(INTDIR)\graphoptimization.obj"
	-@erase "$(INTDIR)\labels.obj"
	-@erase "$(INTDIR)\lvq_pak.obj"
	-@erase "$(INTDIR)\mds.obj"
	-@erase "$(INTDIR)\optimizeAnchors.obj"
	-@erase "$(INTDIR)\orangeom.obj"
	-@erase "$(INTDIR)\som.obj"
	-@erase "$(INTDIR)\som_rout.obj"
	-@erase "$(INTDIR)\triangulate.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\version.obj"
	-@erase "$(INTDIR)\WmlDelaunay2a.obj"
	-@erase "$(INTDIR)\WmlMath.obj"
	-@erase "$(INTDIR)\WmlSystem.obj"
	-@erase "$(INTDIR)\WmlVector2.obj"
	-@erase "$(OUTDIR)\orangeom.exp"
	-@erase "$(OUTDIR)\orangeom.lib"
	-@erase "$(OUTDIR)\orangeom.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

MTL=midl.exe
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangeom.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=orange.lib /nologo /dll /pdb:none /machine:I386 /out:"$(OUTDIR)\orangeom.pyd" /implib:"$(OUTDIR)\orangeom.lib" /libpath:"../../lib" /libpath:"$(PYTHON)\libs" 
LINK32_OBJS= \
	"$(INTDIR)\WmlDelaunay2a.obj" \
	"$(INTDIR)\WmlMath.obj" \
	"$(INTDIR)\WmlSystem.obj" \
	"$(INTDIR)\WmlVector2.obj" \
	"$(INTDIR)\datafile.obj" \
	"$(INTDIR)\fileio.obj" \
	"$(INTDIR)\labels.obj" \
	"$(INTDIR)\lvq_pak.obj" \
	"$(INTDIR)\som_rout.obj" \
	"$(INTDIR)\version.obj" \
	"$(INTDIR)\graphDrawing.obj" \
	"$(INTDIR)\graphoptimization.obj" \
	"$(INTDIR)\mds.obj" \
	"$(INTDIR)\optimizeAnchors.obj" \
	"$(INTDIR)\orangeom.obj" \
	"$(INTDIR)\som.obj" \
	"$(INTDIR)\triangulate.obj"

"$(OUTDIR)\orangeom.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

$(DS_POSTBUILD_DEP) : "$(OUTDIR)\orangeom.pyd"
   ..\upx.bat orangeom
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "orangeom - Win32 Debug"

OUTDIR=.\obj/Debug
INTDIR=.\obj/Debug

ALL : "..\..\orangeom_d.pyd"


CLEAN :
	-@erase "$(INTDIR)\datafile.obj"
	-@erase "$(INTDIR)\fileio.obj"
	-@erase "$(INTDIR)\graphDrawing.obj"
	-@erase "$(INTDIR)\graphoptimization.obj"
	-@erase "$(INTDIR)\labels.obj"
	-@erase "$(INTDIR)\lvq_pak.obj"
	-@erase "$(INTDIR)\mds.obj"
	-@erase "$(INTDIR)\optimizeAnchors.obj"
	-@erase "$(INTDIR)\orangeom.obj"
	-@erase "$(INTDIR)\som.obj"
	-@erase "$(INTDIR)\som_rout.obj"
	-@erase "$(INTDIR)\triangulate.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\version.obj"
	-@erase "$(INTDIR)\WmlDelaunay2a.obj"
	-@erase "$(INTDIR)\WmlMath.obj"
	-@erase "$(INTDIR)\WmlSystem.obj"
	-@erase "$(INTDIR)\WmlVector2.obj"
	-@erase "$(OUTDIR)\orangeom_d.exp"
	-@erase "$(OUTDIR)\orangeom_d.lib"
	-@erase "$(OUTDIR)\orangeom_d.pdb"
	-@erase "..\..\orangeom_d.ilk"
	-@erase "..\..\orangeom_d.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

MTL=midl.exe
MTL_PROJ=/nologo /D "_DEBUG" /mktyplib203 /win32 
RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangeom.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=orange_d.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\orangeom_d.pdb" /debug /machine:I386 /out:"..\..\orangeom_d.pyd" /implib:"$(OUTDIR)\orangeom_d.lib" /pdbtype:sept /libpath:"$(PYTHON)\libs" /libpath:"../../lib" 
LINK32_OBJS= \
	"$(INTDIR)\WmlDelaunay2a.obj" \
	"$(INTDIR)\WmlMath.obj" \
	"$(INTDIR)\WmlSystem.obj" \
	"$(INTDIR)\WmlVector2.obj" \
	"$(INTDIR)\datafile.obj" \
	"$(INTDIR)\fileio.obj" \
	"$(INTDIR)\labels.obj" \
	"$(INTDIR)\lvq_pak.obj" \
	"$(INTDIR)\som_rout.obj" \
	"$(INTDIR)\version.obj" \
	"$(INTDIR)\graphDrawing.obj" \
	"$(INTDIR)\graphoptimization.obj" \
	"$(INTDIR)\mds.obj" \
	"$(INTDIR)\optimizeAnchors.obj" \
	"$(INTDIR)\orangeom.obj" \
	"$(INTDIR)\som.obj" \
	"$(INTDIR)\triangulate.obj"

"..\..\orangeom_d.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

$(DS_POSTBUILD_DEP) : "..\..\orangeom_d.pyd"
   copy obj\Debug\orangeom_d.lib ..\..\lib\orangeom_d.lib
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "orangeom - Win32 Release_Debug"

OUTDIR=.\obj/Release_Debug
INTDIR=.\obj/Release_Debug
# Begin Custom Macros
OutDir=.\obj/Release_Debug
# End Custom Macros

ALL : "$(OUTDIR)\orangeom.pyd"


CLEAN :
	-@erase "$(INTDIR)\datafile.obj"
	-@erase "$(INTDIR)\fileio.obj"
	-@erase "$(INTDIR)\graphDrawing.obj"
	-@erase "$(INTDIR)\graphoptimization.obj"
	-@erase "$(INTDIR)\labels.obj"
	-@erase "$(INTDIR)\lvq_pak.obj"
	-@erase "$(INTDIR)\mds.obj"
	-@erase "$(INTDIR)\optimizeAnchors.obj"
	-@erase "$(INTDIR)\orangeom.obj"
	-@erase "$(INTDIR)\som.obj"
	-@erase "$(INTDIR)\som_rout.obj"
	-@erase "$(INTDIR)\triangulate.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(INTDIR)\version.obj"
	-@erase "$(INTDIR)\WmlDelaunay2a.obj"
	-@erase "$(INTDIR)\WmlMath.obj"
	-@erase "$(INTDIR)\WmlSystem.obj"
	-@erase "$(INTDIR)\WmlVector2.obj"
	-@erase "$(OUTDIR)\orangeom.exp"
	-@erase "$(OUTDIR)\orangeom.lib"
	-@erase "$(OUTDIR)\orangeom.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

.c{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.obj::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.c{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cpp{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

.cxx{$(INTDIR)}.sbr::
   $(CPP) @<<
   $(CPP_PROJ) $< 
<<

MTL=midl.exe
MTL_PROJ=/nologo /D "NDEBUG" /mktyplib203 /win32 
RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangeom.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=orange.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /pdb:none /debug /machine:I386 /out:"$(OUTDIR)\orangeom.pyd" /implib:"$(OUTDIR)\orangeom.lib" /libpath:"$(PYTHON)\libs" /libpath:"../../lib" 
LINK32_OBJS= \
	"$(INTDIR)\WmlDelaunay2a.obj" \
	"$(INTDIR)\WmlMath.obj" \
	"$(INTDIR)\WmlSystem.obj" \
	"$(INTDIR)\WmlVector2.obj" \
	"$(INTDIR)\datafile.obj" \
	"$(INTDIR)\fileio.obj" \
	"$(INTDIR)\labels.obj" \
	"$(INTDIR)\lvq_pak.obj" \
	"$(INTDIR)\som_rout.obj" \
	"$(INTDIR)\version.obj" \
	"$(INTDIR)\graphDrawing.obj" \
	"$(INTDIR)\graphoptimization.obj" \
	"$(INTDIR)\mds.obj" \
	"$(INTDIR)\optimizeAnchors.obj" \
	"$(INTDIR)\orangeom.obj" \
	"$(INTDIR)\som.obj" \
	"$(INTDIR)\triangulate.obj"

"$(OUTDIR)\orangeom.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj/Release_Debug
# End Custom Macros

$(DS_POSTBUILD_DEP) : "$(OUTDIR)\orangeom.pyd"
   del ..\..\orangeom.pyd
	copy "obj\release_debug\orangeom.pyd" "..\..\orangeom.pyd"
	copy obj\Release_debug\orangeom.lib ..\..\lib\orangeom.lib
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("orangeom.dep")
!INCLUDE "orangeom.dep"
!ELSE 
!MESSAGE Warning: cannot find "orangeom.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "orangeom - Win32 Release" || "$(CFG)" == "orangeom - Win32 Debug" || "$(CFG)" == "orangeom - Win32 Release_Debug"
SOURCE=.\wml\WmlDelaunay2a.cpp

"$(INTDIR)\WmlDelaunay2a.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\wml\WmlMath.cpp

"$(INTDIR)\WmlMath.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\wml\WmlSystem.cpp

"$(INTDIR)\WmlSystem.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\wml\WmlVector2.cpp

"$(INTDIR)\WmlVector2.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\datafile.c

"$(INTDIR)\datafile.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\fileio.c

"$(INTDIR)\fileio.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\labels.c

"$(INTDIR)\labels.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\lvq_pak.c

"$(INTDIR)\lvq_pak.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\som_rout.c

"$(INTDIR)\som_rout.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\som\version.c

"$(INTDIR)\version.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\graphDrawing.cpp

"$(INTDIR)\graphDrawing.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\graphoptimization.cpp

"$(INTDIR)\graphoptimization.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\mds.cpp

!IF  "$(CFG)" == "orangeom - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /D "NO_PIPED_COMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\mds.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\mds.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\mds.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\optimizeAnchors.cpp

!IF  "$(CFG)" == "orangeom - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /D "NO_PIPED_COMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\optimizeAnchors.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\optimizeAnchors.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /Zi /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\optimizeAnchors.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\orangeom.cpp

!IF  "$(CFG)" == "orangeom - Win32 Release"

CPP_SWITCHES=/MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /D "NO_PIPED_COMANDS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\orangeom.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\orangeom.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Release_Debug"

CPP_SWITCHES=/MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\orangeom.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\som.cpp

"$(INTDIR)\som.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\triangulate.cpp

!IF  "$(CFG)" == "orangeom - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /D "NO_PIPED_COMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\triangulate.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\triangulate.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "orangeom - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../orange" /I "../include" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGEOM_EXPORTS" /D "NO_PIPED_COMMANDS" /Fp"$(INTDIR)\orangeom.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\triangulate.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 


!ENDIF 

