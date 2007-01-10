# Microsoft Developer Studio Generated NMAKE File, Based on include.dsp
!IF "$(CFG)" == ""
CFG=include - Win32 Debug
!MESSAGE No configuration specified. Defaulting to include - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "include - Win32 Release" && "$(CFG)" != "include - Win32 Debug" && "$(CFG)" != "include - Win32 Release_Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "include.mak" CFG="include - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "include - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "include - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE "include - Win32 Release_Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "include - Win32 Release"

OUTDIR=.\obj/Release
INTDIR=.\obj/Release
# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

ALL : "$(OUTDIR)\include.lib"


CLEAN :
	-@erase "$(INTDIR)\c2py.obj"
	-@erase "$(INTDIR)\common.obj"
	-@erase "$(INTDIR)\crc32.obj"
	-@erase "$(INTDIR)\lcomb.obj"
	-@erase "$(INTDIR)\stat.obj"
	-@erase "$(INTDIR)\statexceptions.obj"
	-@erase "$(INTDIR)\strings.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\include.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /Fp"$(INTDIR)\include.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\include.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\include.lib" 
LIB32_OBJS= \
	"$(INTDIR)\c2py.obj" \
	"$(INTDIR)\common.obj" \
	"$(INTDIR)\crc32.obj" \
	"$(INTDIR)\lcomb.obj" \
	"$(INTDIR)\stat.obj" \
	"$(INTDIR)\statexceptions.obj" \
	"$(INTDIR)\strings.obj"

"$(OUTDIR)\include.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "include - Win32 Debug"

OUTDIR=.\obj\Debug
INTDIR=.\obj\Debug
# Begin Custom Macros
OutDir=.\obj\Debug
# End Custom Macros

ALL : "$(OUTDIR)\include.lib"


CLEAN :
	-@erase "$(INTDIR)\c2py.obj"
	-@erase "$(INTDIR)\common.obj"
	-@erase "$(INTDIR)\crc32.obj"
	-@erase "$(INTDIR)\lcomb.obj"
	-@erase "$(INTDIR)\stat.obj"
	-@erase "$(INTDIR)\statexceptions.obj"
	-@erase "$(INTDIR)\strings.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\include.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\include.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"$(OUTDIR)\include.lib" 
LIB32_OBJS= \
	"$(INTDIR)\c2py.obj" \
	"$(INTDIR)\common.obj" \
	"$(INTDIR)\crc32.obj" \
	"$(INTDIR)\lcomb.obj" \
	"$(INTDIR)\stat.obj" \
	"$(INTDIR)\statexceptions.obj" \
	"$(INTDIR)\strings.obj"

"$(OUTDIR)\include.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ELSEIF  "$(CFG)" == "include - Win32 Release_Debug"

OUTDIR=c:\temp\orange\release_debug
INTDIR=c:\temp\orange\release_debug

ALL : "..\..\lib\include_d.lib"


CLEAN :
	-@erase "$(INTDIR)\c2py.obj"
	-@erase "$(INTDIR)\common.obj"
	-@erase "$(INTDIR)\crc32.obj"
	-@erase "$(INTDIR)\lcomb.obj"
	-@erase "$(INTDIR)\stat.obj"
	-@erase "$(INTDIR)\statexceptions.obj"
	-@erase "$(INTDIR)\strings.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "..\..\lib\include_d.lib"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /Fp"$(INTDIR)\include.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

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

RSC=rc.exe
BSC32=bscmake.exe
BSC32_FLAGS=/nologo /o"$(OUTDIR)\include.bsc" 
BSC32_SBRS= \
	
LIB32=link.exe -lib
LIB32_FLAGS=/nologo /out:"..\..\lib\include_d.lib" 
LIB32_OBJS= \
	"$(INTDIR)\c2py.obj" \
	"$(INTDIR)\common.obj" \
	"$(INTDIR)\crc32.obj" \
	"$(INTDIR)\lcomb.obj" \
	"$(INTDIR)\stat.obj" \
	"$(INTDIR)\statexceptions.obj" \
	"$(INTDIR)\strings.obj"

"..\..\lib\include_d.lib" : "$(OUTDIR)" $(DEF_FILE) $(LIB32_OBJS)
    $(LIB32) @<<
  $(LIB32_FLAGS) $(DEF_FLAGS) $(LIB32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("include.dep")
!INCLUDE "include.dep"
!ELSE 
!MESSAGE Warning: cannot find "include.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "include - Win32 Release" || "$(CFG)" == "include - Win32 Debug" || "$(CFG)" == "include - Win32 Release_Debug"
SOURCE=.\c2py.cpp

"$(INTDIR)\c2py.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\common.cpp

"$(INTDIR)\common.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\crc32.cpp

"$(INTDIR)\crc32.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\lcomb.cpp

"$(INTDIR)\lcomb.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\stat.cpp

!IF  "$(CFG)" == "include - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /Fp"$(INTDIR)\include.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\stat.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "include - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /c 

"$(INTDIR)\stat.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "include - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /Fp"$(INTDIR)\include.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /c 

"$(INTDIR)\stat.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\statexceptions.cpp

"$(INTDIR)\statexceptions.obj" : $(SOURCE) "$(INTDIR)"


SOURCE=.\strings.cpp

"$(INTDIR)\strings.obj" : $(SOURCE) "$(INTDIR)"



!ENDIF 

