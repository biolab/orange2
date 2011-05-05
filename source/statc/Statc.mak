# Microsoft Developer Studio Generated NMAKE File, Based on Statc.dsp
!IF "$(CFG)" == ""
CFG=Statc - Win32 Debug
!MESSAGE No configuration specified. Defaulting to Statc - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "Statc - Win32 Release" && "$(CFG)" != "Statc - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "Statc.mak" CFG="Statc - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "Statc - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "Statc - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "Statc - Win32 Release"

OUTDIR=.\obj/Release
INTDIR=.\obj/Release
# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\statc.pyd"

!ELSE 

ALL : "include - Win32 Release" "$(OUTDIR)\statc.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\stat.obj"
	-@erase "$(INTDIR)\statc.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\statc.exp"
	-@erase "$(OUTDIR)\statc.lib"
	-@erase "$(OUTDIR)\statc.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "STATC_EXPORTS" /Fp"$(INTDIR)\Statc.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Statc.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=/nologo /dll /pdb:none /machine:I386 /out:"$(OUTDIR)\statc.pyd" /implib:"$(OUTDIR)\statc.lib" /libpath:"../../lib" /libpath:"$(PYTHON)\libs" 
LINK32_OBJS= \
	"$(INTDIR)\stat.obj" \
	"$(INTDIR)\statc.obj" \
	"..\include\obj\Release\include.lib"

"$(OUTDIR)\statc.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

$(DS_POSTBUILD_DEP) : "include - Win32 Release" "$(OUTDIR)\statc.pyd"
   ..\upx.bat statc
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "Statc - Win32 Debug"

OUTDIR=.\obj/Debug
INTDIR=.\obj/Debug

!IF "$(RECURSE)" == "0" 

ALL : "..\..\statc_d.pyd"

!ELSE 

ALL : "include - Win32 Debug" "..\..\statc_d.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\stat.obj"
	-@erase "$(INTDIR)\statc.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\statc_d.exp"
	-@erase "$(OUTDIR)\statc_d.lib"
	-@erase "$(OUTDIR)\statc_d.pdb"
	-@erase "..\..\statc_d.ilk"
	-@erase "..\..\statc_d.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../include" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "STATC_EXPORTS" /Fp"$(INTDIR)\Statc.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Statc.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\statc_d.pdb" /debug /machine:I386 /out:"../../statc_d.pyd" /implib:"$(OUTDIR)\statc_d.lib" /pdbtype:sept /libpath:"$(PYTHON)/libs" 
LINK32_OBJS= \
	"$(INTDIR)\stat.obj" \
	"$(INTDIR)\statc.obj" \
	"..\include\obj\Debug\include.lib"

"..\..\statc_d.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("Statc.dep")
!INCLUDE "Statc.dep"
!ELSE 
!MESSAGE Warning: cannot find "Statc.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "Statc - Win32 Release" || "$(CFG)" == "Statc - Win32 Debug"
SOURCE=..\include\stat.cpp

"$(INTDIR)\stat.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


SOURCE=.\statc.cpp

"$(INTDIR)\statc.obj" : $(SOURCE) "$(INTDIR)"


!IF  "$(CFG)" == "Statc - Win32 Release"

"include - Win32 Release" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" 
   cd "..\statc"

"include - Win32 ReleaseCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" RECURSE=1 CLEAN 
   cd "..\statc"

!ELSEIF  "$(CFG)" == "Statc - Win32 Debug"

"include - Win32 Debug" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" 
   cd "..\statc"

"include - Win32 DebugCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" RECURSE=1 CLEAN 
   cd "..\statc"

!ENDIF 


!ENDIF 

