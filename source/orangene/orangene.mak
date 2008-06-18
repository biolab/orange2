# Microsoft Developer Studio Generated NMAKE File, Based on orangene.dsp
!IF "$(CFG)" == ""
CFG=orangene - Win32 Release
!MESSAGE No configuration specified. Defaulting to orangene - Win32 Release.
!ENDIF 

!IF "$(CFG)" != "orangene - Win32 Release" && "$(CFG)" != "orangene - Win32 Debug" && "$(CFG)" != "orangene - Win32 Release_Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "orangene.mak" CFG="orangene - Win32 Release"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "orangene - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "orangene - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "orangene - Win32 Release_Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "orangene - Win32 Release"

OUTDIR=.\obj\release
INTDIR=.\obj\release
# Begin Custom Macros
OutDir=.\obj\release
# End Custom Macros

ALL : "$(OUTDIR)\orangene.pyd"


CLEAN :
	-@erase "$(INTDIR)\heatmap.obj"
	-@erase "$(INTDIR)\orangene.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\orangene.exp"
	-@erase "$(OUTDIR)\orangene.lib"
	-@erase "$(OUTDIR)\orangene.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "../orange" /I "../orange/px" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGENE_EXPORTS" /Fp"$(INTDIR)\orangene.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangene.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=/nologo /dll /pdb:none /machine:I386 /out:"$(OUTDIR)\orangene.pyd" /implib:"$(OUTDIR)\orangene.lib" /libpath:"../../lib" /libpath:"$(PYTHON)\libs" 
LINK32_OBJS= \
	"$(INTDIR)\heatmap.obj" \
	"$(INTDIR)\orangene.obj"

"$(OUTDIR)\orangene.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj\release
# End Custom Macros

$(DS_POSTBUILD_DEP) : "$(OUTDIR)\orangene.pyd"
   ..\upx.bat orangene
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "orangene - Win32 Debug"

OUTDIR=.\obj\debug
INTDIR=.\obj\debug

ALL : "..\..\orangene_d.pyd"


CLEAN :
	-@erase "$(INTDIR)\heatmap.obj"
	-@erase "$(INTDIR)\orangene.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\orangene_d.exp"
	-@erase "$(OUTDIR)\orangene_d.lib"
	-@erase "$(OUTDIR)\orangene_d.pdb"
	-@erase "..\..\orangene_d.ilk"
	-@erase "..\..\orangene_d.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MTd /W3 /Gm /GR /GX /ZI /Od /I "px" /I "../include" /I "$(PYTHON)\include" /I "../orange" /I "../orange/px" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGENE_EXPORTS" /Fp"$(INTDIR)\orangene.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangene.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\orangene_d.pdb" /debug /machine:I386 /out:"..\..\orangene_d.pyd" /implib:"$(OUTDIR)\orangene_d.lib" /pdbtype:sept /libpath:"../../lib" /libpath:"$(PYTHON)/libs" 
LINK32_OBJS= \
	"$(INTDIR)\heatmap.obj" \
	"$(INTDIR)\orangene.obj"

"..\..\orangene_d.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

$(DS_POSTBUILD_DEP) : "..\..\orangene_d.pyd"
   copy obj\Debug\orangene_d.lib ..\..\lib\orangene_d.lib
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "orangene - Win32 Release_Debug"

OUTDIR=.\obj/release_debug
INTDIR=.\obj/release_debug

ALL : "..\..\orangene.pyd"


CLEAN :
	-@erase "$(INTDIR)\heatmap.obj"
	-@erase "$(INTDIR)\orangene.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\orangene.exp"
	-@erase "$(OUTDIR)\orangene.lib"
	-@erase "$(OUTDIR)\orangene.pdb"
	-@erase "..\..\orangene.ilk"
	-@erase "..\..\orangene.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /Zi /Od /I "../include" /I "$(PYTHON)\include" /I "../orange" /I "../orange/px" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGENE_EXPORTS" /Fp"$(INTDIR)\orangene.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\orangene.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\orangene.pdb" /debug /machine:I386 /out:"..\..\orangene.pyd" /implib:"$(OUTDIR)\orangene.lib" /libpath:"../../lib" /libpath:"$(PYTHON)\libs" 
LINK32_OBJS= \
	"$(INTDIR)\heatmap.obj" \
	"$(INTDIR)\orangene.obj"

"..\..\orangene.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

$(DS_POSTBUILD_DEP) : "..\..\orangene.pyd"
   copy obj\Release\orangene.lib ..\..\lib\orangene.lib
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("orangene.dep")
!INCLUDE "orangene.dep"
!ELSE 
!MESSAGE Warning: cannot find "orangene.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "orangene - Win32 Release" || "$(CFG)" == "orangene - Win32 Debug" || "$(CFG)" == "orangene - Win32 Release_Debug"
SOURCE=.\heatmap.cpp

"$(INTDIR)\heatmap.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\orangene.cpp

"$(INTDIR)\orangene.obj" : $(SOURCE) "$(INTDIR)"



!ENDIF 

