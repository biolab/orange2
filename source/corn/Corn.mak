# Microsoft Developer Studio Generated NMAKE File, Based on Corn.dsp
!IF "$(CFG)" == ""
CFG=Corn - Win32 Debug
!MESSAGE No configuration specified. Defaulting to Corn - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "Corn - Win32 Release" && "$(CFG)" != "Corn - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "Corn.mak" CFG="Corn - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "Corn - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "Corn - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "Corn - Win32 Release"

OUTDIR=.\obj/Release
INTDIR=.\obj/Release
# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\corn.pyd"

!ELSE 

ALL : "include - Win32 Release" "$(OUTDIR)\corn.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\corn.obj"
	-@erase "$(INTDIR)\numeric_interface.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\corn.exp"
	-@erase "$(OUTDIR)\corn.lib"
	-@erase "$(OUTDIR)\corn.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "CORN_EXPORTS" /Fp"$(INTDIR)\Corn.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Corn.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=orange.lib /nologo /dll /pdb:none /machine:I386 /out:"$(OUTDIR)\corn.pyd" /implib:"$(OUTDIR)\corn.lib" /libpath:"../../lib" /libpath:"$(PYTHON)\libs" 
LINK32_OBJS= \
	"$(INTDIR)\corn.obj" \
	"$(INTDIR)\numeric_interface.obj" \
	"..\include\obj\Release\include.lib"

"$(OUTDIR)\corn.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

$(DS_POSTBUILD_DEP) : "include - Win32 Release" "$(OUTDIR)\corn.pyd"
   ..\upx.bat corn
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "Corn - Win32 Debug"

OUTDIR=.\obj/Debug
INTDIR=.\obj/Debug

!IF "$(RECURSE)" == "0" 

ALL : "..\..\Corn_d.pyd"

!ELSE 

ALL : "include - Win32 Debug" "..\..\Corn_d.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\corn.obj"
	-@erase "$(INTDIR)\numeric_interface.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\Corn_d.exp"
	-@erase "$(OUTDIR)\Corn_d.lib"
	-@erase "$(OUTDIR)\Corn_d.pdb"
	-@erase "..\..\Corn_d.ilk"
	-@erase "..\..\Corn_d.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../include" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "CORN_EXPORTS" /Fp"$(INTDIR)\Corn.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Corn.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\Corn_d.pdb" /debug /machine:I386 /out:"../../Corn_d.pyd" /implib:"$(OUTDIR)\Corn_d.lib" /pdbtype:sept /libpath:"$(PYTHON)/libs" 
LINK32_OBJS= \
	"$(INTDIR)\corn.obj" \
	"$(INTDIR)\numeric_interface.obj" \
	"..\include\obj\Debug\include.lib"

"..\..\Corn_d.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("Corn.dep")
!INCLUDE "Corn.dep"
!ELSE 
!MESSAGE Warning: cannot find "Corn.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "Corn - Win32 Release" || "$(CFG)" == "Corn - Win32 Debug"
SOURCE=.\corn.cpp

"$(INTDIR)\corn.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=..\orange\numeric_interface.cpp

"$(INTDIR)\numeric_interface.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) $(CPP_PROJ) $(SOURCE)


!IF  "$(CFG)" == "Corn - Win32 Release"

"include - Win32 Release" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" 
   cd "..\corn"

"include - Win32 ReleaseCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" RECURSE=1 CLEAN 
   cd "..\corn"

!ELSEIF  "$(CFG)" == "Corn - Win32 Debug"

"include - Win32 Debug" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" 
   cd "..\corn"

"include - Win32 DebugCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" RECURSE=1 CLEAN 
   cd "..\corn"

!ENDIF 


!ENDIF 

