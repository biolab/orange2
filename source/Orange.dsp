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
# PROP Output_Dir "c:\temp\orange\release"
# PROP Intermediate_Dir "c:\temp\orange\release"
# PROP Ignore_Export_Lib 1
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GR /GX /O2 /I "include" /I "orange/ppp" /I "orange/px" /I "../external" /I "$(PYTHON)\include" /I "$(GNUWIN32)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /YX /FD /Zm700 /c
# SUBTRACT CPP /Fr
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 libgslcblas.a libgsl.a kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /pdb:none /machine:I386 /out:"c:\temp\orange\release\orange.pyd" /libpath:"$(PYTHON)\libs" /libpath:"$(GNUWIN32)\lib" /WARN:0
# SUBTRACT LINK32 /debug
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=UPXing Orange
PostBuild_Cmds=del "d:\ai\orange\orange.pyd"	"c:\program files\upx" "c:\temp\orange\release\orange.pyd" -o "d:\ai\orange\orange.pyd"	rem copy "c:\temp\orange\release\orange.pyd" "d:\ai\orange\orange.pyd"
# End Special Build Tool

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "c:\temp\orange\debug"
# PROP Intermediate_Dir "c:\temp\orange\debug"
# PROP Ignore_Export_Lib 1
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GR /GX /ZI /Od /I "include" /I "orange/ppp" /I "orange/px" /I "$(PYTHON)\include" /I "$(GNUWIN32)/include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /YX /FD /GZ /Zm700 /c
# SUBTRACT CPP /Fr
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 libgslcblas.a libgsl.a kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /out:"c:\temp\orange\debug\orange_d.pyd" /pdbtype:sept /libpath:"$(PYTHON)/libs" /libpath:"$(GNUWIN32)/lib"
# SUBTRACT LINK32 /verbose /nodefaultlib

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Orange___Win32_Release_Debug"
# PROP BASE Intermediate_Dir "Orange___Win32_Release_Debug"
# PROP BASE Ignore_Export_Lib 1
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "c:\temp\orange\release_debug"
# PROP Intermediate_Dir "c:\temp\orange\release_debug"
# PROP Ignore_Export_Lib 1
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GR /GX /O2 /I "include" /I "orange/ppp" /I "orange/px" /I "../external" /I "$(PYTHON)\include" /I "$(GNUWIN32)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /YX /FD /Zm700 /c
# SUBTRACT BASE CPP /Fr
# ADD CPP /nologo /MT /W3 /GR /GX /O2 /I "include" /I "orange/ppp" /I "orange/px" /I "../external" /I "$(PYTHON)\include" /I "$(GNUWIN32)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /YX /FD /Zm700 /c
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
# ADD LINK32 libgslcblas.a libgsl.a kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /pdb:none /debug /machine:I386 /out:"d:\ai\orange\orange.pyd" /libpath:"$(PYTHON)\libs" /libpath:"$(GNUWIN32)\lib" /WARN:0

!ENDIF 

# Begin Target

# Name "Orange - Win32 Release"
# Name "Orange - Win32 Debug"
# Name "Orange - Win32 Release_Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\orange\assistant.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\assoc.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\assoc_sparse.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\basket.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\basstat.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\bayes.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\boolcnt.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\c4.5.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\c45inter.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\calibrate.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\callback.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cartesian.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\clas_gen.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\classfromvar.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\classifier.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_example.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_misc.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_orange.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_value.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\contingency.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\converts.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\cost.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\costwrapper.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\decomposition.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\dictproxy.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\discretize.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\dist_clustering.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\distance.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\distance_dtw.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\distancemap.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\distvars.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\domain.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\domaindepot.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\errors.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\estimateprob.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\exampleclustering.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\examplegen.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\examples.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\excel.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\filegen.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\filter.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\functions.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\garbage_py_manner.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\getarg.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\graph.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\gslconversions.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD CPP /Zi /Od

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\hclust.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\heatmap.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\im_col_assess.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# PROP BASE Exclude_From_Build 1
# PROP Exclude_From_Build 1

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\imputation.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\induce.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\knn.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\learn.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_components.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_io.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_kernel.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD CPP /Zi /Od

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\lib_learner.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_preprocess.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_vectors.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\linreg.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\logfit.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\logistic.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\logreg.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lookup.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lsq.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\lwr.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\majority.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\measures.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\meta.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\minimal_complexity.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\minimal_error.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\module.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\nearest.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\numeric_interface.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\obsolete.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\orange.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\orvector.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\preprocessors.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\progress.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\pythonvars.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\random.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\readdata.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\redundancy.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\retisinter.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\root.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\spec_contingency.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\spec_gen.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\stringvars.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\subsets.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\survival.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\svm.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\symmatrix.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\tabdelim.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

# ADD BASE CPP /O2
# SUBTRACT BASE CPP /Z<none>
# ADD CPP /O2
# SUBTRACT CPP /Z<none>

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\orange\table.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt_split.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt_stop.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\transdomain.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\transval.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\trindex.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\valuelisttemplate.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\values.cpp
# End Source File
# Begin Source File

SOURCE=.\orange\vars.cpp
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\orange\assistant.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\assoc.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\basket.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\basstat.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\bayes.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\boolcnt.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\c4.5.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\c45inter.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\calibrate.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\callback.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cartesian.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\clas_gen.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\classfromvar.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\classify.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_example.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_misc.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_orange.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cls_value.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\contingency.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\converts.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\cost.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\costwrapper.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\decomposition.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\discretize.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\dist_clustering.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\distance.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\distance_dtw.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\distancemap.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\distvars.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\domain.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\domaindepot.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\errors.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\estimateprob.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\exampleclustering.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\examplegen.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\examples.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\filegen.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\filter.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\garbage.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\garbage_py_manner.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\getarg.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\graph.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\gslconversions.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\hclust.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\heatmap.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\imputation.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\induce.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\knn.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\learn.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\lib_kernel.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\linreg.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\logfit.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\logistic.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\lookup.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\lwr.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\majority.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\maptemplates.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\measures.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\meta.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\minimal_complexity.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\minimal_error.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\module.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\nearest.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\numeric_interface.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\orange.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\ormap.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\orvector.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\pqueue_i.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\preprocessors.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\progress.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\pythonvars.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\pyxtract_macros.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\random.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\readdata.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\redundancy.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\relief.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\retisinter.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\root.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\slist.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\spec_contingency.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\spec_gen.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\stringvars.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\student.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\subsets.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\svm.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\symmatrix.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\tabdelim.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\table.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt_split.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\tdidt_stop.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\transdomain.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\transval.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\trindex.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\valuelisttemplate.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\values.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\vars.hpp
# End Source File
# Begin Source File

SOURCE=.\orange\vectortemplates.hpp
# End Source File
# End Group
# Begin Group "ppp files"

# PROP Default_Filter "ppp"
# Begin Source File

SOURCE=.\orange\ppp\assistant.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\assoc.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\basstat.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\bayes.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\bayes_clustering.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\boosting.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\c4.5.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\c45inter.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\callback.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\cartesian.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\clas_gen.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\classfromvar.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\classify.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\contingency.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\cost.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\costwrapper.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\decomp.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\decomposition.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\discretize.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\dist_clustering.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\distance.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\distance_dtw.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\distvars.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\domain.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\estimateprob.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\examplegen.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\examples.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\filegen.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\filter.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\induce.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\knn.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\learn.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\linreg.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\lookup.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\majority.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\measures.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\memory.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\minimal_complexity.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\minimal_error.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\nearest.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\preprocess.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\preprocessors.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\random.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\readdata.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\redundancy.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\relief.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\retisinter.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\root.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\spec_contingency.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\spec_gen.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\stringvars.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\survival.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\svm.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\svm_filtering.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\tabdelim.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\table.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\tdidt.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\tdidt_split.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\tdidt_stop.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\timestamp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\transdomain.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\transval.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\trindex.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\valuefex.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\values.ppp
# End Source File
# Begin Source File

SOURCE=.\orange\ppp\vars.ppp
# End Source File
# End Group
# Begin Group "px files"

# PROP Default_Filter "px"
# Begin Source File

SOURCE=.\orange\px\callback.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\changes.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\cls_example.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\cls_orange.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\cls_value.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\externs.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\functions.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\initialization.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_components.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_io.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_kernel.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_learner.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_preprocess.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\lib_vectors.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\obsolete.px
# End Source File
# Begin Source File

SOURCE=.\orange\px\timestamp
# End Source File
# End Group
# End Target
# End Project
