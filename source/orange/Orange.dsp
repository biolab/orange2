# Microsoft Developer Studio Project File - Name="Orange" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

CFG=Orange - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "Orange.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "Orange.mak" CFG="Orange - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "Orange - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "Orange - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "Orange - Win32 Release_Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "Orange - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "obj/Release"
# PROP Intermediate_Dir "obj/Release"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "px" /I "ppp" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /FD /Zm700 /Gs /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 /nologo /dll /pdb:none /machine:I386 /out:"obj/Release/orange.pyd" /libpath:"$(PYTHON)\libs" /WARN:0
# SUBTRACT LINK32 /verbose /debug
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=UPXing Orange
PostBuild_Cmds=..\upx.bat orange
# End Special Build Tool

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "obj\Debug"
# PROP Intermediate_Dir "obj\Debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MDd /W3 /Gm /GR /GX /Zi /Od /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fr /FD /GZ /Zm700 /c
# SUBTRACT CPP /YX
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 ole32.lib oleaut32.lib /nologo /dll /debug /machine:I386 /out:"c:\d\ai\orange\orange_d.pyd" /libpath:"$(PYTHON)/libs"
# SUBTRACT LINK32 /verbose /pdb:none /nodefaultlib
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Cmds=copy obj\Debug\orange_d.lib ..\..\lib\orange_d.lib
# End Special Build Tool

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Orange___Win32_Release_Debug"
# PROP BASE Intermediate_Dir "Orange___Win32_Release_Debug"
# PROP BASE Ignore_Export_Lib 1
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "obj/Release_debug"
# PROP Intermediate_Dir "obj/Release_debug"
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GR /GX /O2 /I "include" /I "orange/ppp" /I "orange/px" /I "../external" /I "$(PYTHON)\include" /I "$(GNUWIN32)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /YX /FD /Zm700 /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MD /W3 /GR /GX /O2 /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /YX /FD /Zm700 /c
# SUBTRACT CPP /Fr
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 libgslcblas.a libgsl.a kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /pdb:none /machine:I386 /out:"c:\temp\orange\release\orange.pyd" /libpath:"$(PYTHON)\libs" /libpath:"$(GNUWIN32)\lib" /WARN:0
# SUBTRACT BASE LINK32 /debug
# ADD LINK32 oleaut32.lib ole32.lib /nologo /dll /pdb:none /debug /machine:I386 /out:"..\..\orange.pyd" /libpath:"$(PYTHON)\libs" /WARN:0

!ENDIF 

# Begin Target

# Name "Orange - Win32 Release"
# Name "Orange - Win32 Debug"
# Name "Orange - Win32 Release_Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\assistant.cpp
# End Source File
# Begin Source File

SOURCE=.\assoc.cpp
# End Source File
# Begin Source File

SOURCE=.\assoc_sparse.cpp
# End Source File
# Begin Source File

SOURCE=.\basket.cpp
# End Source File
# Begin Source File

SOURCE=.\basstat.cpp
# End Source File
# Begin Source File

SOURCE=.\bayes.cpp
# End Source File
# Begin Source File

SOURCE=.\boolcnt.cpp
# End Source File
# Begin Source File

SOURCE=.\c4.5.cpp
# End Source File
# Begin Source File

SOURCE=.\c45inter.cpp
# End Source File
# Begin Source File

SOURCE=.\calibrate.cpp
# End Source File
# Begin Source File

SOURCE=.\callback.cpp
# End Source File
# Begin Source File

SOURCE=.\cartesian.cpp
# End Source File
# Begin Source File

SOURCE=.\clas_gen.cpp
# End Source File
# Begin Source File

SOURCE=.\classfromvar.cpp
# End Source File
# Begin Source File

SOURCE=.\classifier.cpp
# End Source File
# Begin Source File

SOURCE=.\cls_example.cpp
# End Source File
# Begin Source File

SOURCE=.\cls_misc.cpp
# End Source File
# Begin Source File

SOURCE=.\cls_orange.cpp
# End Source File
# Begin Source File

SOURCE=.\cls_value.cpp
# End Source File
# Begin Source File

SOURCE=.\contingency.cpp
# End Source File
# Begin Source File

SOURCE=.\converts.cpp
# End Source File
# Begin Source File

SOURCE=.\cost.cpp
# End Source File
# Begin Source File

SOURCE=.\costwrapper.cpp
# End Source File
# Begin Source File

SOURCE=.\decomposition.cpp
# End Source File
# Begin Source File

SOURCE=.\dictproxy.cpp
# End Source File
# Begin Source File

SOURCE=.\discretize.cpp
# End Source File
# Begin Source File

SOURCE=.\dist_clustering.cpp
# End Source File
# Begin Source File

SOURCE=.\distance.cpp
# End Source File
# Begin Source File

SOURCE=.\distance_dtw.cpp
# End Source File
# Begin Source File

SOURCE=.\distancemap.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\distvars.cpp
# End Source File
# Begin Source File

SOURCE=.\domain.cpp
# End Source File
# Begin Source File

SOURCE=.\domaindepot.cpp
# End Source File
# Begin Source File

SOURCE=.\errors.cpp
# End Source File
# Begin Source File

SOURCE=.\estimateprob.cpp
# End Source File
# Begin Source File

SOURCE=.\exampleclustering.cpp
# End Source File
# Begin Source File

SOURCE=.\examplegen.cpp
# End Source File
# Begin Source File

SOURCE=.\examples.cpp
# End Source File
# Begin Source File

SOURCE=.\excel.cpp
# End Source File
# Begin Source File

SOURCE=.\filegen.cpp
# End Source File
# Begin Source File

SOURCE=.\filter.cpp
# End Source File
# Begin Source File

SOURCE=.\functions.cpp
# End Source File
# Begin Source File

SOURCE=.\garbage.cpp
# End Source File
# Begin Source File

SOURCE=.\getarg.cpp
# End Source File
# Begin Source File

SOURCE=.\graph.cpp
# End Source File
# Begin Source File

SOURCE=.\gslconversions.cpp
# End Source File
# Begin Source File

SOURCE=.\hclust.cpp
# End Source File
# Begin Source File

SOURCE=.\imputation.cpp
# End Source File
# Begin Source File

SOURCE=.\induce.cpp
# End Source File
# Begin Source File

SOURCE=.\jit_linker.cpp
# End Source File
# Begin Source File

SOURCE=.\knn.cpp
# End Source File
# Begin Source File

SOURCE=.\learn.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_components.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_io.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_kernel.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_learner.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_preprocess.cpp
# End Source File
# Begin Source File

SOURCE=.\lib_vectors.cpp
# End Source File
# Begin Source File

SOURCE=.\linreg.cpp
# End Source File
# Begin Source File

SOURCE=.\logfit.cpp
# End Source File
# Begin Source File

SOURCE=.\logistic.cpp
# End Source File
# Begin Source File

SOURCE=.\logreg.cpp
# End Source File
# Begin Source File

SOURCE=.\lookup.cpp
# End Source File
# Begin Source File

SOURCE=.\lsq.cpp
# End Source File
# Begin Source File

SOURCE=.\lwr.cpp
# End Source File
# Begin Source File

SOURCE=.\majority.cpp
# End Source File
# Begin Source File

SOURCE=.\measures.cpp
# End Source File
# Begin Source File

SOURCE=.\meta.cpp
# End Source File
# Begin Source File

SOURCE=.\minimal_complexity.cpp
# End Source File
# Begin Source File

SOURCE=.\minimal_error.cpp
# End Source File
# Begin Source File

SOURCE=.\nearest.cpp
# End Source File
# Begin Source File

SOURCE=.\numeric_interface.cpp
# End Source File
# Begin Source File

SOURCE=.\orange.cpp
# End Source File
# Begin Source File

SOURCE=.\orvector.cpp
# End Source File
# Begin Source File

SOURCE=.\pnn.cpp
# End Source File
# Begin Source File

SOURCE=.\preprocessors.cpp
# End Source File
# Begin Source File

SOURCE=.\progress.cpp
# End Source File
# Begin Source File

SOURCE=.\pythonvars.cpp
# End Source File
# Begin Source File

SOURCE=.\r_imports.cpp
# End Source File
# Begin Source File

SOURCE=.\random.cpp
# End Source File
# Begin Source File

SOURCE=.\rconversions.cpp
# End Source File
# Begin Source File

SOURCE=.\readdata.cpp
# End Source File
# Begin Source File

SOURCE=.\redundancy.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\retisinter.cpp
# End Source File
# Begin Source File

SOURCE=.\root.cpp
# End Source File
# Begin Source File

SOURCE=.\rulelearner.cpp
# End Source File
# Begin Source File

SOURCE=.\spec_contingency.cpp
# End Source File
# Begin Source File

SOURCE=.\spec_gen.cpp
# End Source File
# Begin Source File

SOURCE=.\stringvars.cpp
# End Source File
# Begin Source File

SOURCE=.\subsets.cpp
# End Source File
# Begin Source File

SOURCE=.\survival.cpp
# End Source File
# Begin Source File

SOURCE=.\svm.cpp
# End Source File
# Begin Source File

SOURCE=.\symmatrix.cpp
# End Source File
# Begin Source File

SOURCE=.\tabdelim.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\table.cpp
# End Source File
# Begin Source File

SOURCE=.\tdidt.cpp
# End Source File
# Begin Source File

SOURCE=.\tdidt_split.cpp
# End Source File
# Begin Source File

SOURCE=.\tdidt_stop.cpp
# End Source File
# Begin Source File

SOURCE=.\transdomain.cpp
# End Source File
# Begin Source File

SOURCE=.\transval.cpp
# End Source File
# Begin Source File

SOURCE=.\trindex.cpp
# End Source File
# Begin Source File

SOURCE=.\valuelisttemplate.cpp
# End Source File
# Begin Source File

SOURCE=.\values.cpp
# End Source File
# Begin Source File

SOURCE=.\vars.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\assistant.hpp
# End Source File
# Begin Source File

SOURCE=.\assoc.hpp
# End Source File
# Begin Source File

SOURCE=.\basket.hpp
# End Source File
# Begin Source File

SOURCE=.\basstat.hpp
# End Source File
# Begin Source File

SOURCE=.\bayes.hpp
# End Source File
# Begin Source File

SOURCE=.\boolcnt.hpp
# End Source File
# Begin Source File

SOURCE=.\c4.5.hpp
# End Source File
# Begin Source File

SOURCE=.\c45inter.hpp
# End Source File
# Begin Source File

SOURCE=.\calibrate.hpp
# End Source File
# Begin Source File

SOURCE=.\callback.hpp
# End Source File
# Begin Source File

SOURCE=.\cartesian.hpp
# End Source File
# Begin Source File

SOURCE=.\clas_gen.hpp
# End Source File
# Begin Source File

SOURCE=.\classfromvar.hpp
# End Source File
# Begin Source File

SOURCE=.\classify.hpp
# End Source File
# Begin Source File

SOURCE=.\cls_example.hpp
# End Source File
# Begin Source File

SOURCE=.\cls_misc.hpp
# End Source File
# Begin Source File

SOURCE=.\cls_orange.hpp
# End Source File
# Begin Source File

SOURCE=.\cls_value.hpp
# End Source File
# Begin Source File

SOURCE=.\contingency.hpp
# End Source File
# Begin Source File

SOURCE=.\converts.hpp
# End Source File
# Begin Source File

SOURCE=.\cost.hpp
# End Source File
# Begin Source File

SOURCE=.\costwrapper.hpp
# End Source File
# Begin Source File

SOURCE=.\decomposition.hpp
# End Source File
# Begin Source File

SOURCE=.\discretize.hpp
# End Source File
# Begin Source File

SOURCE=.\dist_clustering.hpp
# End Source File
# Begin Source File

SOURCE=.\distance.hpp
# End Source File
# Begin Source File

SOURCE=.\distance_dtw.hpp
# End Source File
# Begin Source File

SOURCE=.\distancemap.hpp
# End Source File
# Begin Source File

SOURCE=.\distvars.hpp
# End Source File
# Begin Source File

SOURCE=.\domain.hpp
# End Source File
# Begin Source File

SOURCE=.\domaindepot.hpp
# End Source File
# Begin Source File

SOURCE=.\errors.hpp
# End Source File
# Begin Source File

SOURCE=.\estimateprob.hpp
# End Source File
# Begin Source File

SOURCE=.\exampleclustering.hpp
# End Source File
# Begin Source File

SOURCE=.\examplegen.hpp
# End Source File
# Begin Source File

SOURCE=.\examples.hpp
# End Source File
# Begin Source File

SOURCE=.\filegen.hpp
# End Source File
# Begin Source File

SOURCE=.\filter.hpp
# End Source File
# Begin Source File

SOURCE=.\garbage.hpp
# End Source File
# Begin Source File

SOURCE=.\getarg.hpp
# End Source File
# Begin Source File

SOURCE=.\graph.hpp
# End Source File
# Begin Source File

SOURCE=.\gslconversions.hpp
# End Source File
# Begin Source File

SOURCE=.\hclust.hpp
# End Source File
# Begin Source File

SOURCE=.\imputation.hpp
# End Source File
# Begin Source File

SOURCE=.\induce.hpp
# End Source File
# Begin Source File

SOURCE=.\jit_linker.hpp
# End Source File
# Begin Source File

SOURCE=.\knn.hpp
# End Source File
# Begin Source File

SOURCE=.\learn.hpp
# End Source File
# Begin Source File

SOURCE=.\lib_kernel.hpp
# End Source File
# Begin Source File

SOURCE=.\linreg.hpp
# End Source File
# Begin Source File

SOURCE=.\logfit.hpp
# End Source File
# Begin Source File

SOURCE=.\logistic.hpp
# End Source File
# Begin Source File

SOURCE=.\lookup.hpp
# End Source File
# Begin Source File

SOURCE=.\lwr.hpp
# End Source File
# Begin Source File

SOURCE=.\majority.hpp
# End Source File
# Begin Source File

SOURCE=.\maptemplates.hpp
# End Source File
# Begin Source File

SOURCE=.\measures.hpp
# End Source File
# Begin Source File

SOURCE=.\meta.hpp
# End Source File
# Begin Source File

SOURCE=.\minimal_complexity.hpp
# End Source File
# Begin Source File

SOURCE=.\minimal_error.hpp
# End Source File
# Begin Source File

SOURCE=.\nearest.hpp
# End Source File
# Begin Source File

SOURCE=.\numeric_interface.hpp
# End Source File
# Begin Source File

SOURCE=.\orange.hpp
# End Source File
# Begin Source File

SOURCE=.\orange_api.hpp
# End Source File
# Begin Source File

SOURCE=.\ormap.hpp
# End Source File
# Begin Source File

SOURCE=.\orvector.hpp
# End Source File
# Begin Source File

SOURCE=.\pnn.hpp
# End Source File
# Begin Source File

SOURCE=.\pqueue_i.hpp
# End Source File
# Begin Source File

SOURCE=.\preprocessors.hpp
# End Source File
# Begin Source File

SOURCE=.\progress.hpp
# End Source File
# Begin Source File

SOURCE=.\pythonvars.hpp
# End Source File
# Begin Source File

SOURCE=.\pyxtract_macros.hpp
# End Source File
# Begin Source File

SOURCE=.\random.hpp
# End Source File
# Begin Source File

SOURCE=.\redundancy.hpp
# End Source File
# Begin Source File

SOURCE=.\relief.hpp
# End Source File
# Begin Source File

SOURCE=.\retisinter.hpp
# End Source File
# Begin Source File

SOURCE=.\root.hpp
# End Source File
# Begin Source File

SOURCE=.\rulelearner.hpp
# End Source File
# Begin Source File

SOURCE=.\slist.hpp
# End Source File
# Begin Source File

SOURCE=.\spec_contingency.hpp
# End Source File
# Begin Source File

SOURCE=.\spec_gen.hpp
# End Source File
# Begin Source File

SOURCE=.\stringvars.hpp
# End Source File
# Begin Source File

SOURCE=.\subsets.hpp
# End Source File
# Begin Source File

SOURCE=.\svm.hpp
# End Source File
# Begin Source File

SOURCE=.\symmatrix.hpp
# End Source File
# Begin Source File

SOURCE=.\tabdelim.hpp
# End Source File
# Begin Source File

SOURCE=.\table.hpp
# End Source File
# Begin Source File

SOURCE=.\tdidt.hpp
# End Source File
# Begin Source File

SOURCE=.\tdidt_split.hpp
# End Source File
# Begin Source File

SOURCE=.\tdidt_stop.hpp
# End Source File
# Begin Source File

SOURCE=.\transdomain.hpp
# End Source File
# Begin Source File

SOURCE=.\transval.hpp
# End Source File
# Begin Source File

SOURCE=.\trindex.hpp
# End Source File
# Begin Source File

SOURCE=.\valuelisttemplate.hpp
# End Source File
# Begin Source File

SOURCE=.\values.hpp
# End Source File
# Begin Source File

SOURCE=.\vars.hpp
# End Source File
# Begin Source File

SOURCE=.\vectortemplates.hpp
# End Source File
# End Group
# Begin Group "ppp files"

# PROP Default_Filter "ppp"
# Begin Source File

SOURCE=.\ppp\assistant.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\assoc.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\basstat.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\bayes.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\bayes_clustering.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\boosting.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\c4.5.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\c45inter.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\callback.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\cartesian.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\clas_gen.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\classfromvar.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\classify.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\contingency.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\cost.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\costwrapper.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\decomp.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\decomposition.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\discretize.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\dist_clustering.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\distance.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\distance_dtw.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\distvars.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\domain.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\estimateprob.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\examplegen.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\examples.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\filegen.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\filter.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\imputation.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\induce.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\knn.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\learn.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\linreg.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\lookup.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\majority.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\measures.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\memory.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\minimal_complexity.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\minimal_error.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\nearest.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\preprocess.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\preprocessors.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\random.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\readdata.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\redundancy.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\relief.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\retisinter.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\root.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\spec_contingency.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\spec_gen.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\stringvars.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\survival.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\svm.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\svm_filtering.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\tabdelim.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\table.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\tdidt.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\tdidt_split.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\tdidt_stop.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\timestamp
# End Source File
# Begin Source File

SOURCE=.\ppp\transdomain.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\transval.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\trindex.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\valuefex.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\values.ppp
# End Source File
# Begin Source File

SOURCE=.\ppp\vars.ppp
# End Source File
# End Group
# Begin Group "px files"

# PROP Default_Filter "px"
# Begin Source File

SOURCE=.\px\cls_example.px
# End Source File
# Begin Source File

SOURCE=.\px\cls_orange.px
# End Source File
# Begin Source File

SOURCE=.\px\cls_value.px
# End Source File
# Begin Source File

SOURCE=.\px\externs.px
# End Source File
# Begin Source File

SOURCE=.\px\initialization.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_components.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_io.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_kernel.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_learner.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_preprocess.px
# End Source File
# Begin Source File

SOURCE=.\px\lib_vectors.px
# End Source File
# Begin Source File

SOURCE=.\px\orange.px
# End Source File
# Begin Source File

SOURCE=.\px\timestamp
# End Source File
# End Group
# End Target
# End Project
