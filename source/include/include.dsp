# Microsoft Developer Studio Project File - Name="include" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=include - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "include.mak".
!MESSAGE 
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

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "include - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "obj/Release"
# PROP Intermediate_Dir "obj/Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "include - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "obj\Debug"
# PROP Intermediate_Dir "obj\Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /FD /GZ /c
# SUBTRACT CPP /Fr /YX /Yc /Yu
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "include - Win32 Release_Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "include___Win32_Release_Debug"
# PROP BASE Intermediate_Dir "include___Win32_Release_Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "c:\temp\orange\release_debug"
# PROP Intermediate_Dir "c:\temp\orange\release_debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# SUBTRACT CPP /Fr
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"..\..\lib\include_d.lib"

!ENDIF 

# Begin Target

# Name "include - Win32 Release"
# Name "include - Win32 Debug"
# Name "include - Win32 Release_Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\c2py.cpp
# End Source File
# Begin Source File

SOURCE=.\common.cpp
# End Source File
# Begin Source File

SOURCE=.\crc32.cpp
# End Source File
# Begin Source File

SOURCE=.\lcomb.cpp
# End Source File
# Begin Source File

SOURCE=.\stat.cpp

!IF  "$(CFG)" == "include - Win32 Release"

# ADD CPP /O2

!ELSEIF  "$(CFG)" == "include - Win32 Debug"

!ELSEIF  "$(CFG)" == "include - Win32 Release_Debug"

# ADD BASE CPP /O2
# ADD CPP /O2

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\statexceptions.cpp
# End Source File
# Begin Source File

SOURCE=.\strings.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\c2py.hpp
# End Source File
# Begin Source File

SOURCE=.\cMersenneTwister.h
# End Source File
# Begin Source File

SOURCE=.\crc.h
# End Source File
# Begin Source File

SOURCE=.\lcomb.hpp
# End Source File
# Begin Source File

SOURCE=.\stat.hpp
# End Source File
# Begin Source File

SOURCE=.\statexceptions.hpp
# End Source File
# Begin Source File

SOURCE=.\stladdon.hpp
# End Source File
# Begin Source File

SOURCE=.\strings.hpp
# End Source File
# End Group
# End Target
# End Project
