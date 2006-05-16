# Microsoft Developer Studio Project File - Name="mymodule" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=mymodule - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "mymodule.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "mymodule.mak" CFG="mymodule - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "mymodule - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "mymodule - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "mymodule - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "MYMODULE_EXPORTS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I "../include" /I "../orange" /I "px" /I "ppp" /I "$(PYTHON)/include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "CORE_EXPORTS" /YX /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386 /out:"../../core.pyd" /libpath:"../../lib" /libpath:"$(PYTHON)/libs"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy Release\mymodule.lib ..\..\lib\mymodule.lib
# End Special Build Tool

!ELSEIF  "$(CFG)" == "mymodule - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "MYMODULE_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GR /GX /ZI /Od /I "../include" /I "../orange" /I "px" /I "ppp" /I "$(PYTHON)/include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "CORE_EXPORTS" /YX /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /out:"../../core_d.pyd" /pdbtype:sept /libpath:"../../lib" /libpath:"$(PYTHON)/libs"
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy Debug\mymodule_d.lib ..\..\lib\mymodule_d.lib
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "mymodule - Win32 Release"
# Name "mymodule - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\binarize.cpp
# End Source File
# Begin Source File

SOURCE=.\binnode.cpp
# End Source File
# Begin Source File

SOURCE=.\binnode.h
# End Source File
# Begin Source File

SOURCE=.\binpart.cpp
# End Source File
# Begin Source File

SOURCE=.\binpart.h
# End Source File
# Begin Source File

SOURCE=.\bintree.cpp
# End Source File
# Begin Source File

SOURCE=.\bintree.h
# End Source File
# Begin Source File

SOURCE=.\bool.h
# End Source File
# Begin Source File

SOURCE=.\C5convert.cpp
# End Source File
# Begin Source File

SOURCE=.\C5defns.h
# End Source File
# Begin Source File

SOURCE=.\C5hooks.cpp
# End Source File
# Begin Source File

SOURCE=.\C5hooks.h
# End Source File
# Begin Source File

SOURCE=.\cls_myclasses.cpp
# End Source File
# Begin Source File

SOURCE=.\constrct.cpp
# End Source File
# Begin Source File

SOURCE=.\constrct.h
# End Source File
# Begin Source File

SOURCE=.\contain.h
# End Source File
# Begin Source File

SOURCE=.\cost.cpp
# End Source File
# Begin Source File

SOURCE=.\dectree.cpp
# End Source File
# Begin Source File

SOURCE=.\dectree.h
# End Source File
# Begin Source File

SOURCE=.\error.cpp
# End Source File
# Begin Source File

SOURCE=.\error.h
# End Source File
# Begin Source File

SOURCE=.\estimator.cpp
# End Source File
# Begin Source File

SOURCE=.\estimator.h
# End Source File
# Begin Source File

SOURCE=.\estOrdAttr.cpp
# End Source File
# Begin Source File

SOURCE=.\expr.cpp
# End Source File
# Begin Source File

SOURCE=.\expr.h
# End Source File
# Begin Source File

SOURCE=.\frontend.cpp
# End Source File
# Begin Source File

SOURCE=.\frontend.h
# End Source File
# Begin Source File

SOURCE=.\ftree.cpp
# End Source File
# Begin Source File

SOURCE=.\ftree.h
# End Source File
# Begin Source File

SOURCE=.\general.h
# End Source File
# Begin Source File

SOURCE=.\mathutil.cpp
# End Source File
# Begin Source File

SOURCE=.\mathutil.h
# End Source File
# Begin Source File

SOURCE=.\menu.cpp
# End Source File
# Begin Source File

SOURCE=.\menu.h
# End Source File
# Begin Source File

SOURCE=.\model.cpp
# End Source File
# Begin Source File

SOURCE=.\myclasses.cpp
# PROP Exclude_From_Build 1
# End Source File
# Begin Source File

SOURCE=.\mymodule.cpp
# End Source File
# Begin Source File

SOURCE=.\new_new.cpp
# End Source File
# Begin Source File

SOURCE=.\new_new.h
# End Source File
# Begin Source File

SOURCE=.\nrutil.cpp
# End Source File
# Begin Source File

SOURCE=.\nrutil.h
# End Source File
# Begin Source File

SOURCE=.\options.cpp
# End Source File
# Begin Source File

SOURCE=.\options.h
# End Source File
# Begin Source File

SOURCE=.\prune.cpp
# End Source File
# Begin Source File

SOURCE=.\randomForestClass.cpp
# End Source File
# Begin Source File

SOURCE=.\relieff.cpp
# End Source File
# Begin Source File

SOURCE=.\rfRegularize.cpp
# End Source File
# Begin Source File

SOURCE=.\rfUtil.cpp
# End Source File
# Begin Source File

SOURCE=.\rfUtil.h
# End Source File
# Begin Source File

SOURCE=.\rndforest.cpp
# End Source File
# Begin Source File

SOURCE=.\rndforest.h
# End Source File
# Begin Source File

SOURCE=.\treenode.cpp
# End Source File
# Begin Source File

SOURCE=.\trutil.cpp
# End Source File
# Begin Source File

SOURCE=.\utils.cpp
# End Source File
# Begin Source File

SOURCE=.\utils.h
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\myclasses.hpp
# End Source File
# Begin Source File

SOURCE=.\mymodule_globals.hpp
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# End Group
# End Target
# End Project
