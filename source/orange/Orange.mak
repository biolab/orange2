# Microsoft Developer Studio Generated NMAKE File, Based on Orange.dsp
!IF "$(CFG)" == ""
CFG=Orange - Win32 Debug
!MESSAGE No configuration specified. Defaulting to Orange - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "Orange - Win32 Release" && "$(CFG)" != "Orange - Win32 Debug" && "$(CFG)" != "Orange - Win32 Release_Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
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
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 

!IF  "$(CFG)" == "Orange - Win32 Release"

OUTDIR=.\obj/Release
INTDIR=.\obj/Release
# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "$(OUTDIR)\orange.pyd"

!ELSE 

ALL : "include - Win32 Release" "$(OUTDIR)\orange.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 ReleaseCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\assistant.obj"
	-@erase "$(INTDIR)\assoc.obj"
	-@erase "$(INTDIR)\assoc_sparse.obj"
	-@erase "$(INTDIR)\basket.obj"
	-@erase "$(INTDIR)\basstat.obj"
	-@erase "$(INTDIR)\bayes.obj"
	-@erase "$(INTDIR)\boolcnt.obj"
	-@erase "$(INTDIR)\c4.5.obj"
	-@erase "$(INTDIR)\c45inter.obj"
	-@erase "$(INTDIR)\calibrate.obj"
	-@erase "$(INTDIR)\callback.obj"
	-@erase "$(INTDIR)\cartesian.obj"
	-@erase "$(INTDIR)\clas_gen.obj"
	-@erase "$(INTDIR)\classfromvar.obj"
	-@erase "$(INTDIR)\classifier.obj"
	-@erase "$(INTDIR)\cls_example.obj"
	-@erase "$(INTDIR)\cls_misc.obj"
	-@erase "$(INTDIR)\cls_orange.obj"
	-@erase "$(INTDIR)\cls_value.obj"
	-@erase "$(INTDIR)\contingency.obj"
	-@erase "$(INTDIR)\converts.obj"
	-@erase "$(INTDIR)\cost.obj"
	-@erase "$(INTDIR)\costwrapper.obj"
	-@erase "$(INTDIR)\decomposition.obj"
	-@erase "$(INTDIR)\dictproxy.obj"
	-@erase "$(INTDIR)\discretize.obj"
	-@erase "$(INTDIR)\dist_clustering.obj"
	-@erase "$(INTDIR)\distance.obj"
	-@erase "$(INTDIR)\distance_dtw.obj"
	-@erase "$(INTDIR)\distancemap.obj"
	-@erase "$(INTDIR)\distvars.obj"
	-@erase "$(INTDIR)\domain.obj"
	-@erase "$(INTDIR)\domaindepot.obj"
	-@erase "$(INTDIR)\errors.obj"
	-@erase "$(INTDIR)\estimateprob.obj"
	-@erase "$(INTDIR)\exampleclustering.obj"
	-@erase "$(INTDIR)\examplegen.obj"
	-@erase "$(INTDIR)\examples.obj"
	-@erase "$(INTDIR)\excel.obj"
	-@erase "$(INTDIR)\filegen.obj"
	-@erase "$(INTDIR)\filter.obj"
	-@erase "$(INTDIR)\functions.obj"
	-@erase "$(INTDIR)\garbage.obj"
	-@erase "$(INTDIR)\getarg.obj"
	-@erase "$(INTDIR)\graph.obj"
	-@erase "$(INTDIR)\gslconversions.obj"
	-@erase "$(INTDIR)\hclust.obj"
	-@erase "$(INTDIR)\imputation.obj"
	-@erase "$(INTDIR)\induce.obj"
	-@erase "$(INTDIR)\jit_linker.obj"
	-@erase "$(INTDIR)\knn.obj"
	-@erase "$(INTDIR)\learn.obj"
	-@erase "$(INTDIR)\lib_components.obj"
	-@erase "$(INTDIR)\lib_io.obj"
	-@erase "$(INTDIR)\lib_kernel.obj"
	-@erase "$(INTDIR)\lib_learner.obj"
	-@erase "$(INTDIR)\lib_preprocess.obj"
	-@erase "$(INTDIR)\lib_vectors.obj"
	-@erase "$(INTDIR)\linreg.obj"
	-@erase "$(INTDIR)\logfit.obj"
	-@erase "$(INTDIR)\logistic.obj"
	-@erase "$(INTDIR)\logreg.obj"
	-@erase "$(INTDIR)\lookup.obj"
	-@erase "$(INTDIR)\lsq.obj"
	-@erase "$(INTDIR)\lwr.obj"
	-@erase "$(INTDIR)\majority.obj"
	-@erase "$(INTDIR)\measures.obj"
	-@erase "$(INTDIR)\meta.obj"
	-@erase "$(INTDIR)\minimal_complexity.obj"
	-@erase "$(INTDIR)\minimal_error.obj"
	-@erase "$(INTDIR)\nearest.obj"
	-@erase "$(INTDIR)\numeric_interface.obj"
	-@erase "$(INTDIR)\orange.obj"
	-@erase "$(INTDIR)\orvector.obj"
	-@erase "$(INTDIR)\pnn.obj"
	-@erase "$(INTDIR)\preprocessors.obj"
	-@erase "$(INTDIR)\progress.obj"
	-@erase "$(INTDIR)\pythonvars.obj"
	-@erase "$(INTDIR)\r_imports.obj"
	-@erase "$(INTDIR)\random.obj"
	-@erase "$(INTDIR)\rconversions.obj"
	-@erase "$(INTDIR)\readdata.obj"
	-@erase "$(INTDIR)\redundancy.obj"
	-@erase "$(INTDIR)\retisinter.obj"
	-@erase "$(INTDIR)\root.obj"
	-@erase "$(INTDIR)\rulelearner.obj"
	-@erase "$(INTDIR)\spec_contingency.obj"
	-@erase "$(INTDIR)\spec_gen.obj"
	-@erase "$(INTDIR)\stringvars.obj"
	-@erase "$(INTDIR)\subsets.obj"
	-@erase "$(INTDIR)\survival.obj"
	-@erase "$(INTDIR)\svm.obj"
	-@erase "$(INTDIR)\symmatrix.obj"
	-@erase "$(INTDIR)\tabdelim.obj"
	-@erase "$(INTDIR)\table.obj"
	-@erase "$(INTDIR)\tdidt.obj"
	-@erase "$(INTDIR)\tdidt_split.obj"
	-@erase "$(INTDIR)\tdidt_stop.obj"
	-@erase "$(INTDIR)\transdomain.obj"
	-@erase "$(INTDIR)\transval.obj"
	-@erase "$(INTDIR)\trindex.obj"
	-@erase "$(INTDIR)\valuelisttemplate.obj"
	-@erase "$(INTDIR)\values.obj"
	-@erase "$(INTDIR)\vars.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\orange.exp"
	-@erase "$(OUTDIR)\orange.lib"
	-@erase "$(OUTDIR)\orange.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "px" /I "ppp" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /Gs /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Orange.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=/nologo /dll /pdb:none /machine:I386 /out:"$(OUTDIR)\orange.pyd" /implib:"$(OUTDIR)\orange.lib" /libpath:"$(PYTHON)\libs" /WARN:0 
LINK32_OBJS= \
	"$(INTDIR)\assistant.obj" \
	"$(INTDIR)\assoc.obj" \
	"$(INTDIR)\assoc_sparse.obj" \
	"$(INTDIR)\basket.obj" \
	"$(INTDIR)\basstat.obj" \
	"$(INTDIR)\bayes.obj" \
	"$(INTDIR)\boolcnt.obj" \
	"$(INTDIR)\c4.5.obj" \
	"$(INTDIR)\c45inter.obj" \
	"$(INTDIR)\calibrate.obj" \
	"$(INTDIR)\callback.obj" \
	"$(INTDIR)\cartesian.obj" \
	"$(INTDIR)\clas_gen.obj" \
	"$(INTDIR)\classfromvar.obj" \
	"$(INTDIR)\classifier.obj" \
	"$(INTDIR)\cls_example.obj" \
	"$(INTDIR)\cls_misc.obj" \
	"$(INTDIR)\cls_orange.obj" \
	"$(INTDIR)\cls_value.obj" \
	"$(INTDIR)\contingency.obj" \
	"$(INTDIR)\converts.obj" \
	"$(INTDIR)\cost.obj" \
	"$(INTDIR)\costwrapper.obj" \
	"$(INTDIR)\decomposition.obj" \
	"$(INTDIR)\dictproxy.obj" \
	"$(INTDIR)\discretize.obj" \
	"$(INTDIR)\dist_clustering.obj" \
	"$(INTDIR)\distance.obj" \
	"$(INTDIR)\distance_dtw.obj" \
	"$(INTDIR)\distancemap.obj" \
	"$(INTDIR)\distvars.obj" \
	"$(INTDIR)\domain.obj" \
	"$(INTDIR)\domaindepot.obj" \
	"$(INTDIR)\errors.obj" \
	"$(INTDIR)\estimateprob.obj" \
	"$(INTDIR)\exampleclustering.obj" \
	"$(INTDIR)\examplegen.obj" \
	"$(INTDIR)\examples.obj" \
	"$(INTDIR)\excel.obj" \
	"$(INTDIR)\filegen.obj" \
	"$(INTDIR)\filter.obj" \
	"$(INTDIR)\functions.obj" \
	"$(INTDIR)\garbage.obj" \
	"$(INTDIR)\getarg.obj" \
	"$(INTDIR)\graph.obj" \
	"$(INTDIR)\gslconversions.obj" \
	"$(INTDIR)\hclust.obj" \
	"$(INTDIR)\imputation.obj" \
	"$(INTDIR)\induce.obj" \
	"$(INTDIR)\jit_linker.obj" \
	"$(INTDIR)\knn.obj" \
	"$(INTDIR)\learn.obj" \
	"$(INTDIR)\lib_components.obj" \
	"$(INTDIR)\lib_io.obj" \
	"$(INTDIR)\lib_kernel.obj" \
	"$(INTDIR)\lib_learner.obj" \
	"$(INTDIR)\lib_preprocess.obj" \
	"$(INTDIR)\lib_vectors.obj" \
	"$(INTDIR)\linreg.obj" \
	"$(INTDIR)\logfit.obj" \
	"$(INTDIR)\logistic.obj" \
	"$(INTDIR)\logreg.obj" \
	"$(INTDIR)\lookup.obj" \
	"$(INTDIR)\lsq.obj" \
	"$(INTDIR)\lwr.obj" \
	"$(INTDIR)\majority.obj" \
	"$(INTDIR)\measures.obj" \
	"$(INTDIR)\meta.obj" \
	"$(INTDIR)\minimal_complexity.obj" \
	"$(INTDIR)\minimal_error.obj" \
	"$(INTDIR)\nearest.obj" \
	"$(INTDIR)\numeric_interface.obj" \
	"$(INTDIR)\orange.obj" \
	"$(INTDIR)\orvector.obj" \
	"$(INTDIR)\pnn.obj" \
	"$(INTDIR)\preprocessors.obj" \
	"$(INTDIR)\progress.obj" \
	"$(INTDIR)\pythonvars.obj" \
	"$(INTDIR)\r_imports.obj" \
	"$(INTDIR)\random.obj" \
	"$(INTDIR)\rconversions.obj" \
	"$(INTDIR)\readdata.obj" \
	"$(INTDIR)\redundancy.obj" \
	"$(INTDIR)\retisinter.obj" \
	"$(INTDIR)\root.obj" \
	"$(INTDIR)\rulelearner.obj" \
	"$(INTDIR)\spec_contingency.obj" \
	"$(INTDIR)\spec_gen.obj" \
	"$(INTDIR)\stringvars.obj" \
	"$(INTDIR)\subsets.obj" \
	"$(INTDIR)\survival.obj" \
	"$(INTDIR)\svm.obj" \
	"$(INTDIR)\symmatrix.obj" \
	"$(INTDIR)\tabdelim.obj" \
	"$(INTDIR)\table.obj" \
	"$(INTDIR)\tdidt.obj" \
	"$(INTDIR)\tdidt_split.obj" \
	"$(INTDIR)\tdidt_stop.obj" \
	"$(INTDIR)\transdomain.obj" \
	"$(INTDIR)\transval.obj" \
	"$(INTDIR)\trindex.obj" \
	"$(INTDIR)\valuelisttemplate.obj" \
	"$(INTDIR)\values.obj" \
	"$(INTDIR)\vars.obj" \
	"..\include\obj\Release\include.lib"

"$(OUTDIR)\orange.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
PostBuild_Desc=UPXing Orange
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj/Release
# End Custom Macros

$(DS_POSTBUILD_DEP) : "include - Win32 Release" "$(OUTDIR)\orange.pyd"
   ..\upx.bat orange
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

OUTDIR=.\obj\Debug
INTDIR=.\obj\Debug
# Begin Custom Macros
OutDir=.\obj\Debug
# End Custom Macros

!IF "$(RECURSE)" == "0" 

ALL : "..\..\orange_d.pyd" "$(OUTDIR)\Orange.bsc"

!ELSE 

ALL : "include - Win32 Debug" "..\..\orange_d.pyd" "$(OUTDIR)\Orange.bsc"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\assistant.obj"
	-@erase "$(INTDIR)\assistant.sbr"
	-@erase "$(INTDIR)\assoc.obj"
	-@erase "$(INTDIR)\assoc.sbr"
	-@erase "$(INTDIR)\assoc_sparse.obj"
	-@erase "$(INTDIR)\assoc_sparse.sbr"
	-@erase "$(INTDIR)\basket.obj"
	-@erase "$(INTDIR)\basket.sbr"
	-@erase "$(INTDIR)\basstat.obj"
	-@erase "$(INTDIR)\basstat.sbr"
	-@erase "$(INTDIR)\bayes.obj"
	-@erase "$(INTDIR)\bayes.sbr"
	-@erase "$(INTDIR)\boolcnt.obj"
	-@erase "$(INTDIR)\boolcnt.sbr"
	-@erase "$(INTDIR)\c4.5.obj"
	-@erase "$(INTDIR)\c4.5.sbr"
	-@erase "$(INTDIR)\c45inter.obj"
	-@erase "$(INTDIR)\c45inter.sbr"
	-@erase "$(INTDIR)\calibrate.obj"
	-@erase "$(INTDIR)\calibrate.sbr"
	-@erase "$(INTDIR)\callback.obj"
	-@erase "$(INTDIR)\callback.sbr"
	-@erase "$(INTDIR)\cartesian.obj"
	-@erase "$(INTDIR)\cartesian.sbr"
	-@erase "$(INTDIR)\clas_gen.obj"
	-@erase "$(INTDIR)\clas_gen.sbr"
	-@erase "$(INTDIR)\classfromvar.obj"
	-@erase "$(INTDIR)\classfromvar.sbr"
	-@erase "$(INTDIR)\classifier.obj"
	-@erase "$(INTDIR)\classifier.sbr"
	-@erase "$(INTDIR)\cls_example.obj"
	-@erase "$(INTDIR)\cls_example.sbr"
	-@erase "$(INTDIR)\cls_misc.obj"
	-@erase "$(INTDIR)\cls_misc.sbr"
	-@erase "$(INTDIR)\cls_orange.obj"
	-@erase "$(INTDIR)\cls_orange.sbr"
	-@erase "$(INTDIR)\cls_value.obj"
	-@erase "$(INTDIR)\cls_value.sbr"
	-@erase "$(INTDIR)\contingency.obj"
	-@erase "$(INTDIR)\contingency.sbr"
	-@erase "$(INTDIR)\converts.obj"
	-@erase "$(INTDIR)\converts.sbr"
	-@erase "$(INTDIR)\cost.obj"
	-@erase "$(INTDIR)\cost.sbr"
	-@erase "$(INTDIR)\costwrapper.obj"
	-@erase "$(INTDIR)\costwrapper.sbr"
	-@erase "$(INTDIR)\decomposition.obj"
	-@erase "$(INTDIR)\decomposition.sbr"
	-@erase "$(INTDIR)\dictproxy.obj"
	-@erase "$(INTDIR)\dictproxy.sbr"
	-@erase "$(INTDIR)\discretize.obj"
	-@erase "$(INTDIR)\discretize.sbr"
	-@erase "$(INTDIR)\dist_clustering.obj"
	-@erase "$(INTDIR)\dist_clustering.sbr"
	-@erase "$(INTDIR)\distance.obj"
	-@erase "$(INTDIR)\distance.sbr"
	-@erase "$(INTDIR)\distance_dtw.obj"
	-@erase "$(INTDIR)\distance_dtw.sbr"
	-@erase "$(INTDIR)\distancemap.obj"
	-@erase "$(INTDIR)\distancemap.sbr"
	-@erase "$(INTDIR)\distvars.obj"
	-@erase "$(INTDIR)\distvars.sbr"
	-@erase "$(INTDIR)\domain.obj"
	-@erase "$(INTDIR)\domain.sbr"
	-@erase "$(INTDIR)\domaindepot.obj"
	-@erase "$(INTDIR)\domaindepot.sbr"
	-@erase "$(INTDIR)\errors.obj"
	-@erase "$(INTDIR)\errors.sbr"
	-@erase "$(INTDIR)\estimateprob.obj"
	-@erase "$(INTDIR)\estimateprob.sbr"
	-@erase "$(INTDIR)\exampleclustering.obj"
	-@erase "$(INTDIR)\exampleclustering.sbr"
	-@erase "$(INTDIR)\examplegen.obj"
	-@erase "$(INTDIR)\examplegen.sbr"
	-@erase "$(INTDIR)\examples.obj"
	-@erase "$(INTDIR)\examples.sbr"
	-@erase "$(INTDIR)\excel.obj"
	-@erase "$(INTDIR)\excel.sbr"
	-@erase "$(INTDIR)\filegen.obj"
	-@erase "$(INTDIR)\filegen.sbr"
	-@erase "$(INTDIR)\filter.obj"
	-@erase "$(INTDIR)\filter.sbr"
	-@erase "$(INTDIR)\functions.obj"
	-@erase "$(INTDIR)\functions.sbr"
	-@erase "$(INTDIR)\garbage.obj"
	-@erase "$(INTDIR)\garbage.sbr"
	-@erase "$(INTDIR)\getarg.obj"
	-@erase "$(INTDIR)\getarg.sbr"
	-@erase "$(INTDIR)\graph.obj"
	-@erase "$(INTDIR)\graph.sbr"
	-@erase "$(INTDIR)\gslconversions.obj"
	-@erase "$(INTDIR)\gslconversions.sbr"
	-@erase "$(INTDIR)\hclust.obj"
	-@erase "$(INTDIR)\hclust.sbr"
	-@erase "$(INTDIR)\imputation.obj"
	-@erase "$(INTDIR)\imputation.sbr"
	-@erase "$(INTDIR)\induce.obj"
	-@erase "$(INTDIR)\induce.sbr"
	-@erase "$(INTDIR)\jit_linker.obj"
	-@erase "$(INTDIR)\jit_linker.sbr"
	-@erase "$(INTDIR)\knn.obj"
	-@erase "$(INTDIR)\knn.sbr"
	-@erase "$(INTDIR)\learn.obj"
	-@erase "$(INTDIR)\learn.sbr"
	-@erase "$(INTDIR)\lib_components.obj"
	-@erase "$(INTDIR)\lib_components.sbr"
	-@erase "$(INTDIR)\lib_io.obj"
	-@erase "$(INTDIR)\lib_io.sbr"
	-@erase "$(INTDIR)\lib_kernel.obj"
	-@erase "$(INTDIR)\lib_kernel.sbr"
	-@erase "$(INTDIR)\lib_learner.obj"
	-@erase "$(INTDIR)\lib_learner.sbr"
	-@erase "$(INTDIR)\lib_preprocess.obj"
	-@erase "$(INTDIR)\lib_preprocess.sbr"
	-@erase "$(INTDIR)\lib_vectors.obj"
	-@erase "$(INTDIR)\lib_vectors.sbr"
	-@erase "$(INTDIR)\linreg.obj"
	-@erase "$(INTDIR)\linreg.sbr"
	-@erase "$(INTDIR)\logfit.obj"
	-@erase "$(INTDIR)\logfit.sbr"
	-@erase "$(INTDIR)\logistic.obj"
	-@erase "$(INTDIR)\logistic.sbr"
	-@erase "$(INTDIR)\logreg.obj"
	-@erase "$(INTDIR)\logreg.sbr"
	-@erase "$(INTDIR)\lookup.obj"
	-@erase "$(INTDIR)\lookup.sbr"
	-@erase "$(INTDIR)\lsq.obj"
	-@erase "$(INTDIR)\lsq.sbr"
	-@erase "$(INTDIR)\lwr.obj"
	-@erase "$(INTDIR)\lwr.sbr"
	-@erase "$(INTDIR)\majority.obj"
	-@erase "$(INTDIR)\majority.sbr"
	-@erase "$(INTDIR)\measures.obj"
	-@erase "$(INTDIR)\measures.sbr"
	-@erase "$(INTDIR)\meta.obj"
	-@erase "$(INTDIR)\meta.sbr"
	-@erase "$(INTDIR)\minimal_complexity.obj"
	-@erase "$(INTDIR)\minimal_complexity.sbr"
	-@erase "$(INTDIR)\minimal_error.obj"
	-@erase "$(INTDIR)\minimal_error.sbr"
	-@erase "$(INTDIR)\nearest.obj"
	-@erase "$(INTDIR)\nearest.sbr"
	-@erase "$(INTDIR)\numeric_interface.obj"
	-@erase "$(INTDIR)\numeric_interface.sbr"
	-@erase "$(INTDIR)\orange.obj"
	-@erase "$(INTDIR)\orange.sbr"
	-@erase "$(INTDIR)\orvector.obj"
	-@erase "$(INTDIR)\orvector.sbr"
	-@erase "$(INTDIR)\pnn.obj"
	-@erase "$(INTDIR)\pnn.sbr"
	-@erase "$(INTDIR)\preprocessors.obj"
	-@erase "$(INTDIR)\preprocessors.sbr"
	-@erase "$(INTDIR)\progress.obj"
	-@erase "$(INTDIR)\progress.sbr"
	-@erase "$(INTDIR)\pythonvars.obj"
	-@erase "$(INTDIR)\pythonvars.sbr"
	-@erase "$(INTDIR)\r_imports.obj"
	-@erase "$(INTDIR)\r_imports.sbr"
	-@erase "$(INTDIR)\random.obj"
	-@erase "$(INTDIR)\random.sbr"
	-@erase "$(INTDIR)\rconversions.obj"
	-@erase "$(INTDIR)\rconversions.sbr"
	-@erase "$(INTDIR)\readdata.obj"
	-@erase "$(INTDIR)\readdata.sbr"
	-@erase "$(INTDIR)\redundancy.obj"
	-@erase "$(INTDIR)\redundancy.sbr"
	-@erase "$(INTDIR)\retisinter.obj"
	-@erase "$(INTDIR)\retisinter.sbr"
	-@erase "$(INTDIR)\root.obj"
	-@erase "$(INTDIR)\root.sbr"
	-@erase "$(INTDIR)\rulelearner.obj"
	-@erase "$(INTDIR)\rulelearner.sbr"
	-@erase "$(INTDIR)\spec_contingency.obj"
	-@erase "$(INTDIR)\spec_contingency.sbr"
	-@erase "$(INTDIR)\spec_gen.obj"
	-@erase "$(INTDIR)\spec_gen.sbr"
	-@erase "$(INTDIR)\stringvars.obj"
	-@erase "$(INTDIR)\stringvars.sbr"
	-@erase "$(INTDIR)\subsets.obj"
	-@erase "$(INTDIR)\subsets.sbr"
	-@erase "$(INTDIR)\survival.obj"
	-@erase "$(INTDIR)\survival.sbr"
	-@erase "$(INTDIR)\svm.obj"
	-@erase "$(INTDIR)\svm.sbr"
	-@erase "$(INTDIR)\symmatrix.obj"
	-@erase "$(INTDIR)\symmatrix.sbr"
	-@erase "$(INTDIR)\tabdelim.obj"
	-@erase "$(INTDIR)\tabdelim.sbr"
	-@erase "$(INTDIR)\table.obj"
	-@erase "$(INTDIR)\table.sbr"
	-@erase "$(INTDIR)\tdidt.obj"
	-@erase "$(INTDIR)\tdidt.sbr"
	-@erase "$(INTDIR)\tdidt_split.obj"
	-@erase "$(INTDIR)\tdidt_split.sbr"
	-@erase "$(INTDIR)\tdidt_stop.obj"
	-@erase "$(INTDIR)\tdidt_stop.sbr"
	-@erase "$(INTDIR)\transdomain.obj"
	-@erase "$(INTDIR)\transdomain.sbr"
	-@erase "$(INTDIR)\transval.obj"
	-@erase "$(INTDIR)\transval.sbr"
	-@erase "$(INTDIR)\trindex.obj"
	-@erase "$(INTDIR)\trindex.sbr"
	-@erase "$(INTDIR)\valuelisttemplate.obj"
	-@erase "$(INTDIR)\valuelisttemplate.sbr"
	-@erase "$(INTDIR)\values.obj"
	-@erase "$(INTDIR)\values.sbr"
	-@erase "$(INTDIR)\vars.obj"
	-@erase "$(INTDIR)\vars.sbr"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(INTDIR)\vc60.pdb"
	-@erase "$(OUTDIR)\Orange.bsc"
	-@erase "$(OUTDIR)\orange_d.exp"
	-@erase "$(OUTDIR)\orange_d.lib"
	-@erase "$(OUTDIR)\orange_d.pdb"
	-@erase "..\..\orange_d.ilk"
	-@erase "..\..\orange_d.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MDd /W3 /Gm /GR /GX /Zi /Od /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fr"$(INTDIR)\\" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Orange.bsc" 
BSC32_SBRS= \
	"$(INTDIR)\assistant.sbr" \
	"$(INTDIR)\assoc.sbr" \
	"$(INTDIR)\assoc_sparse.sbr" \
	"$(INTDIR)\basket.sbr" \
	"$(INTDIR)\basstat.sbr" \
	"$(INTDIR)\bayes.sbr" \
	"$(INTDIR)\boolcnt.sbr" \
	"$(INTDIR)\c4.5.sbr" \
	"$(INTDIR)\c45inter.sbr" \
	"$(INTDIR)\calibrate.sbr" \
	"$(INTDIR)\callback.sbr" \
	"$(INTDIR)\cartesian.sbr" \
	"$(INTDIR)\clas_gen.sbr" \
	"$(INTDIR)\classfromvar.sbr" \
	"$(INTDIR)\classifier.sbr" \
	"$(INTDIR)\cls_example.sbr" \
	"$(INTDIR)\cls_misc.sbr" \
	"$(INTDIR)\cls_orange.sbr" \
	"$(INTDIR)\cls_value.sbr" \
	"$(INTDIR)\contingency.sbr" \
	"$(INTDIR)\converts.sbr" \
	"$(INTDIR)\cost.sbr" \
	"$(INTDIR)\costwrapper.sbr" \
	"$(INTDIR)\decomposition.sbr" \
	"$(INTDIR)\dictproxy.sbr" \
	"$(INTDIR)\discretize.sbr" \
	"$(INTDIR)\dist_clustering.sbr" \
	"$(INTDIR)\distance.sbr" \
	"$(INTDIR)\distance_dtw.sbr" \
	"$(INTDIR)\distancemap.sbr" \
	"$(INTDIR)\distvars.sbr" \
	"$(INTDIR)\domain.sbr" \
	"$(INTDIR)\domaindepot.sbr" \
	"$(INTDIR)\errors.sbr" \
	"$(INTDIR)\estimateprob.sbr" \
	"$(INTDIR)\exampleclustering.sbr" \
	"$(INTDIR)\examplegen.sbr" \
	"$(INTDIR)\examples.sbr" \
	"$(INTDIR)\excel.sbr" \
	"$(INTDIR)\filegen.sbr" \
	"$(INTDIR)\filter.sbr" \
	"$(INTDIR)\functions.sbr" \
	"$(INTDIR)\garbage.sbr" \
	"$(INTDIR)\getarg.sbr" \
	"$(INTDIR)\graph.sbr" \
	"$(INTDIR)\gslconversions.sbr" \
	"$(INTDIR)\hclust.sbr" \
	"$(INTDIR)\imputation.sbr" \
	"$(INTDIR)\induce.sbr" \
	"$(INTDIR)\jit_linker.sbr" \
	"$(INTDIR)\knn.sbr" \
	"$(INTDIR)\learn.sbr" \
	"$(INTDIR)\lib_components.sbr" \
	"$(INTDIR)\lib_io.sbr" \
	"$(INTDIR)\lib_kernel.sbr" \
	"$(INTDIR)\lib_learner.sbr" \
	"$(INTDIR)\lib_preprocess.sbr" \
	"$(INTDIR)\lib_vectors.sbr" \
	"$(INTDIR)\linreg.sbr" \
	"$(INTDIR)\logfit.sbr" \
	"$(INTDIR)\logistic.sbr" \
	"$(INTDIR)\logreg.sbr" \
	"$(INTDIR)\lookup.sbr" \
	"$(INTDIR)\lsq.sbr" \
	"$(INTDIR)\lwr.sbr" \
	"$(INTDIR)\majority.sbr" \
	"$(INTDIR)\measures.sbr" \
	"$(INTDIR)\meta.sbr" \
	"$(INTDIR)\minimal_complexity.sbr" \
	"$(INTDIR)\minimal_error.sbr" \
	"$(INTDIR)\nearest.sbr" \
	"$(INTDIR)\numeric_interface.sbr" \
	"$(INTDIR)\orange.sbr" \
	"$(INTDIR)\orvector.sbr" \
	"$(INTDIR)\pnn.sbr" \
	"$(INTDIR)\preprocessors.sbr" \
	"$(INTDIR)\progress.sbr" \
	"$(INTDIR)\pythonvars.sbr" \
	"$(INTDIR)\r_imports.sbr" \
	"$(INTDIR)\random.sbr" \
	"$(INTDIR)\rconversions.sbr" \
	"$(INTDIR)\readdata.sbr" \
	"$(INTDIR)\redundancy.sbr" \
	"$(INTDIR)\retisinter.sbr" \
	"$(INTDIR)\root.sbr" \
	"$(INTDIR)\rulelearner.sbr" \
	"$(INTDIR)\spec_contingency.sbr" \
	"$(INTDIR)\spec_gen.sbr" \
	"$(INTDIR)\stringvars.sbr" \
	"$(INTDIR)\subsets.sbr" \
	"$(INTDIR)\survival.sbr" \
	"$(INTDIR)\svm.sbr" \
	"$(INTDIR)\symmatrix.sbr" \
	"$(INTDIR)\tabdelim.sbr" \
	"$(INTDIR)\table.sbr" \
	"$(INTDIR)\tdidt.sbr" \
	"$(INTDIR)\tdidt_split.sbr" \
	"$(INTDIR)\tdidt_stop.sbr" \
	"$(INTDIR)\transdomain.sbr" \
	"$(INTDIR)\transval.sbr" \
	"$(INTDIR)\trindex.sbr" \
	"$(INTDIR)\valuelisttemplate.sbr" \
	"$(INTDIR)\values.sbr" \
	"$(INTDIR)\vars.sbr"

"$(OUTDIR)\Orange.bsc" : "$(OUTDIR)" $(BSC32_SBRS)
    $(BSC32) @<<
  $(BSC32_FLAGS) $(BSC32_SBRS)
<<

LINK32=link.exe
LINK32_FLAGS=ole32.lib oleaut32.lib /nologo /dll /incremental:yes /pdb:"$(OUTDIR)\orange_d.pdb" /debug /machine:I386 /out:"c:\d\ai\orange\orange_d.pyd" /implib:"$(OUTDIR)\orange_d.lib" /libpath:"$(PYTHON)/libs" 
LINK32_OBJS= \
	"$(INTDIR)\assistant.obj" \
	"$(INTDIR)\assoc.obj" \
	"$(INTDIR)\assoc_sparse.obj" \
	"$(INTDIR)\basket.obj" \
	"$(INTDIR)\basstat.obj" \
	"$(INTDIR)\bayes.obj" \
	"$(INTDIR)\boolcnt.obj" \
	"$(INTDIR)\c4.5.obj" \
	"$(INTDIR)\c45inter.obj" \
	"$(INTDIR)\calibrate.obj" \
	"$(INTDIR)\callback.obj" \
	"$(INTDIR)\cartesian.obj" \
	"$(INTDIR)\clas_gen.obj" \
	"$(INTDIR)\classfromvar.obj" \
	"$(INTDIR)\classifier.obj" \
	"$(INTDIR)\cls_example.obj" \
	"$(INTDIR)\cls_misc.obj" \
	"$(INTDIR)\cls_orange.obj" \
	"$(INTDIR)\cls_value.obj" \
	"$(INTDIR)\contingency.obj" \
	"$(INTDIR)\converts.obj" \
	"$(INTDIR)\cost.obj" \
	"$(INTDIR)\costwrapper.obj" \
	"$(INTDIR)\decomposition.obj" \
	"$(INTDIR)\dictproxy.obj" \
	"$(INTDIR)\discretize.obj" \
	"$(INTDIR)\dist_clustering.obj" \
	"$(INTDIR)\distance.obj" \
	"$(INTDIR)\distance_dtw.obj" \
	"$(INTDIR)\distancemap.obj" \
	"$(INTDIR)\distvars.obj" \
	"$(INTDIR)\domain.obj" \
	"$(INTDIR)\domaindepot.obj" \
	"$(INTDIR)\errors.obj" \
	"$(INTDIR)\estimateprob.obj" \
	"$(INTDIR)\exampleclustering.obj" \
	"$(INTDIR)\examplegen.obj" \
	"$(INTDIR)\examples.obj" \
	"$(INTDIR)\excel.obj" \
	"$(INTDIR)\filegen.obj" \
	"$(INTDIR)\filter.obj" \
	"$(INTDIR)\functions.obj" \
	"$(INTDIR)\garbage.obj" \
	"$(INTDIR)\getarg.obj" \
	"$(INTDIR)\graph.obj" \
	"$(INTDIR)\gslconversions.obj" \
	"$(INTDIR)\hclust.obj" \
	"$(INTDIR)\imputation.obj" \
	"$(INTDIR)\induce.obj" \
	"$(INTDIR)\jit_linker.obj" \
	"$(INTDIR)\knn.obj" \
	"$(INTDIR)\learn.obj" \
	"$(INTDIR)\lib_components.obj" \
	"$(INTDIR)\lib_io.obj" \
	"$(INTDIR)\lib_kernel.obj" \
	"$(INTDIR)\lib_learner.obj" \
	"$(INTDIR)\lib_preprocess.obj" \
	"$(INTDIR)\lib_vectors.obj" \
	"$(INTDIR)\linreg.obj" \
	"$(INTDIR)\logfit.obj" \
	"$(INTDIR)\logistic.obj" \
	"$(INTDIR)\logreg.obj" \
	"$(INTDIR)\lookup.obj" \
	"$(INTDIR)\lsq.obj" \
	"$(INTDIR)\lwr.obj" \
	"$(INTDIR)\majority.obj" \
	"$(INTDIR)\measures.obj" \
	"$(INTDIR)\meta.obj" \
	"$(INTDIR)\minimal_complexity.obj" \
	"$(INTDIR)\minimal_error.obj" \
	"$(INTDIR)\nearest.obj" \
	"$(INTDIR)\numeric_interface.obj" \
	"$(INTDIR)\orange.obj" \
	"$(INTDIR)\orvector.obj" \
	"$(INTDIR)\pnn.obj" \
	"$(INTDIR)\preprocessors.obj" \
	"$(INTDIR)\progress.obj" \
	"$(INTDIR)\pythonvars.obj" \
	"$(INTDIR)\r_imports.obj" \
	"$(INTDIR)\random.obj" \
	"$(INTDIR)\rconversions.obj" \
	"$(INTDIR)\readdata.obj" \
	"$(INTDIR)\redundancy.obj" \
	"$(INTDIR)\retisinter.obj" \
	"$(INTDIR)\root.obj" \
	"$(INTDIR)\rulelearner.obj" \
	"$(INTDIR)\spec_contingency.obj" \
	"$(INTDIR)\spec_gen.obj" \
	"$(INTDIR)\stringvars.obj" \
	"$(INTDIR)\subsets.obj" \
	"$(INTDIR)\survival.obj" \
	"$(INTDIR)\svm.obj" \
	"$(INTDIR)\symmatrix.obj" \
	"$(INTDIR)\tabdelim.obj" \
	"$(INTDIR)\table.obj" \
	"$(INTDIR)\tdidt.obj" \
	"$(INTDIR)\tdidt_split.obj" \
	"$(INTDIR)\tdidt_stop.obj" \
	"$(INTDIR)\transdomain.obj" \
	"$(INTDIR)\transval.obj" \
	"$(INTDIR)\trindex.obj" \
	"$(INTDIR)\valuelisttemplate.obj" \
	"$(INTDIR)\values.obj" \
	"$(INTDIR)\vars.obj" \
	"..\include\obj\Debug\include.lib"

"..\..\orange_d.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

SOURCE="$(InputPath)"
DS_POSTBUILD_DEP=$(INTDIR)\postbld.dep

ALL : $(DS_POSTBUILD_DEP)

# Begin Custom Macros
OutDir=.\obj\Debug
# End Custom Macros

$(DS_POSTBUILD_DEP) : "include - Win32 Debug" "..\..\orange_d.pyd" "$(OUTDIR)\Orange.bsc"
   copy obj\Debug\orange_d.lib ..\..\lib\orange_d.lib
	echo Helper for Post-build step > "$(DS_POSTBUILD_DEP)"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

OUTDIR=.\obj/Release_debug
INTDIR=.\obj/Release_debug

!IF "$(RECURSE)" == "0" 

ALL : "..\..\orange.pyd"

!ELSE 

ALL : "include - Win32 Release_Debug" "..\..\orange.pyd"

!ENDIF 

!IF "$(RECURSE)" == "1" 
CLEAN :"include - Win32 Release_DebugCLEAN" 
!ELSE 
CLEAN :
!ENDIF 
	-@erase "$(INTDIR)\assistant.obj"
	-@erase "$(INTDIR)\assoc.obj"
	-@erase "$(INTDIR)\assoc_sparse.obj"
	-@erase "$(INTDIR)\basket.obj"
	-@erase "$(INTDIR)\basstat.obj"
	-@erase "$(INTDIR)\bayes.obj"
	-@erase "$(INTDIR)\boolcnt.obj"
	-@erase "$(INTDIR)\c4.5.obj"
	-@erase "$(INTDIR)\c45inter.obj"
	-@erase "$(INTDIR)\calibrate.obj"
	-@erase "$(INTDIR)\callback.obj"
	-@erase "$(INTDIR)\cartesian.obj"
	-@erase "$(INTDIR)\clas_gen.obj"
	-@erase "$(INTDIR)\classfromvar.obj"
	-@erase "$(INTDIR)\classifier.obj"
	-@erase "$(INTDIR)\cls_example.obj"
	-@erase "$(INTDIR)\cls_misc.obj"
	-@erase "$(INTDIR)\cls_orange.obj"
	-@erase "$(INTDIR)\cls_value.obj"
	-@erase "$(INTDIR)\contingency.obj"
	-@erase "$(INTDIR)\converts.obj"
	-@erase "$(INTDIR)\cost.obj"
	-@erase "$(INTDIR)\costwrapper.obj"
	-@erase "$(INTDIR)\decomposition.obj"
	-@erase "$(INTDIR)\dictproxy.obj"
	-@erase "$(INTDIR)\discretize.obj"
	-@erase "$(INTDIR)\dist_clustering.obj"
	-@erase "$(INTDIR)\distance.obj"
	-@erase "$(INTDIR)\distance_dtw.obj"
	-@erase "$(INTDIR)\distancemap.obj"
	-@erase "$(INTDIR)\distvars.obj"
	-@erase "$(INTDIR)\domain.obj"
	-@erase "$(INTDIR)\domaindepot.obj"
	-@erase "$(INTDIR)\errors.obj"
	-@erase "$(INTDIR)\estimateprob.obj"
	-@erase "$(INTDIR)\exampleclustering.obj"
	-@erase "$(INTDIR)\examplegen.obj"
	-@erase "$(INTDIR)\examples.obj"
	-@erase "$(INTDIR)\excel.obj"
	-@erase "$(INTDIR)\filegen.obj"
	-@erase "$(INTDIR)\filter.obj"
	-@erase "$(INTDIR)\functions.obj"
	-@erase "$(INTDIR)\garbage.obj"
	-@erase "$(INTDIR)\getarg.obj"
	-@erase "$(INTDIR)\graph.obj"
	-@erase "$(INTDIR)\gslconversions.obj"
	-@erase "$(INTDIR)\hclust.obj"
	-@erase "$(INTDIR)\imputation.obj"
	-@erase "$(INTDIR)\induce.obj"
	-@erase "$(INTDIR)\jit_linker.obj"
	-@erase "$(INTDIR)\knn.obj"
	-@erase "$(INTDIR)\learn.obj"
	-@erase "$(INTDIR)\lib_components.obj"
	-@erase "$(INTDIR)\lib_io.obj"
	-@erase "$(INTDIR)\lib_kernel.obj"
	-@erase "$(INTDIR)\lib_learner.obj"
	-@erase "$(INTDIR)\lib_preprocess.obj"
	-@erase "$(INTDIR)\lib_vectors.obj"
	-@erase "$(INTDIR)\linreg.obj"
	-@erase "$(INTDIR)\logfit.obj"
	-@erase "$(INTDIR)\logistic.obj"
	-@erase "$(INTDIR)\logreg.obj"
	-@erase "$(INTDIR)\lookup.obj"
	-@erase "$(INTDIR)\lsq.obj"
	-@erase "$(INTDIR)\lwr.obj"
	-@erase "$(INTDIR)\majority.obj"
	-@erase "$(INTDIR)\measures.obj"
	-@erase "$(INTDIR)\meta.obj"
	-@erase "$(INTDIR)\minimal_complexity.obj"
	-@erase "$(INTDIR)\minimal_error.obj"
	-@erase "$(INTDIR)\nearest.obj"
	-@erase "$(INTDIR)\numeric_interface.obj"
	-@erase "$(INTDIR)\orange.obj"
	-@erase "$(INTDIR)\orvector.obj"
	-@erase "$(INTDIR)\pnn.obj"
	-@erase "$(INTDIR)\preprocessors.obj"
	-@erase "$(INTDIR)\progress.obj"
	-@erase "$(INTDIR)\pythonvars.obj"
	-@erase "$(INTDIR)\r_imports.obj"
	-@erase "$(INTDIR)\random.obj"
	-@erase "$(INTDIR)\rconversions.obj"
	-@erase "$(INTDIR)\readdata.obj"
	-@erase "$(INTDIR)\redundancy.obj"
	-@erase "$(INTDIR)\retisinter.obj"
	-@erase "$(INTDIR)\root.obj"
	-@erase "$(INTDIR)\rulelearner.obj"
	-@erase "$(INTDIR)\spec_contingency.obj"
	-@erase "$(INTDIR)\spec_gen.obj"
	-@erase "$(INTDIR)\stringvars.obj"
	-@erase "$(INTDIR)\subsets.obj"
	-@erase "$(INTDIR)\survival.obj"
	-@erase "$(INTDIR)\svm.obj"
	-@erase "$(INTDIR)\symmatrix.obj"
	-@erase "$(INTDIR)\tabdelim.obj"
	-@erase "$(INTDIR)\table.obj"
	-@erase "$(INTDIR)\tdidt.obj"
	-@erase "$(INTDIR)\tdidt_split.obj"
	-@erase "$(INTDIR)\tdidt_stop.obj"
	-@erase "$(INTDIR)\transdomain.obj"
	-@erase "$(INTDIR)\transval.obj"
	-@erase "$(INTDIR)\trindex.obj"
	-@erase "$(INTDIR)\valuelisttemplate.obj"
	-@erase "$(INTDIR)\values.obj"
	-@erase "$(INTDIR)\vars.obj"
	-@erase "$(INTDIR)\vc60.idb"
	-@erase "$(OUTDIR)\orange.exp"
	-@erase "$(OUTDIR)\orange.lib"
	-@erase "..\..\orange.pyd"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

CPP=cl.exe
CPP_PROJ=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /Fp"$(INTDIR)\Orange.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

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
BSC32_FLAGS=/nologo /o"$(OUTDIR)\Orange.bsc" 
BSC32_SBRS= \
	
LINK32=link.exe
LINK32_FLAGS=oleaut32.lib ole32.lib /nologo /dll /pdb:none /debug /machine:I386 /out:"..\..\orange.pyd" /implib:"$(OUTDIR)\orange.lib" /libpath:"$(PYTHON)\libs" /WARN:0 
LINK32_OBJS= \
	"$(INTDIR)\assistant.obj" \
	"$(INTDIR)\assoc.obj" \
	"$(INTDIR)\assoc_sparse.obj" \
	"$(INTDIR)\basket.obj" \
	"$(INTDIR)\basstat.obj" \
	"$(INTDIR)\bayes.obj" \
	"$(INTDIR)\boolcnt.obj" \
	"$(INTDIR)\c4.5.obj" \
	"$(INTDIR)\c45inter.obj" \
	"$(INTDIR)\calibrate.obj" \
	"$(INTDIR)\callback.obj" \
	"$(INTDIR)\cartesian.obj" \
	"$(INTDIR)\clas_gen.obj" \
	"$(INTDIR)\classfromvar.obj" \
	"$(INTDIR)\classifier.obj" \
	"$(INTDIR)\cls_example.obj" \
	"$(INTDIR)\cls_misc.obj" \
	"$(INTDIR)\cls_orange.obj" \
	"$(INTDIR)\cls_value.obj" \
	"$(INTDIR)\contingency.obj" \
	"$(INTDIR)\converts.obj" \
	"$(INTDIR)\cost.obj" \
	"$(INTDIR)\costwrapper.obj" \
	"$(INTDIR)\decomposition.obj" \
	"$(INTDIR)\dictproxy.obj" \
	"$(INTDIR)\discretize.obj" \
	"$(INTDIR)\dist_clustering.obj" \
	"$(INTDIR)\distance.obj" \
	"$(INTDIR)\distance_dtw.obj" \
	"$(INTDIR)\distancemap.obj" \
	"$(INTDIR)\distvars.obj" \
	"$(INTDIR)\domain.obj" \
	"$(INTDIR)\domaindepot.obj" \
	"$(INTDIR)\errors.obj" \
	"$(INTDIR)\estimateprob.obj" \
	"$(INTDIR)\exampleclustering.obj" \
	"$(INTDIR)\examplegen.obj" \
	"$(INTDIR)\examples.obj" \
	"$(INTDIR)\excel.obj" \
	"$(INTDIR)\filegen.obj" \
	"$(INTDIR)\filter.obj" \
	"$(INTDIR)\functions.obj" \
	"$(INTDIR)\garbage.obj" \
	"$(INTDIR)\getarg.obj" \
	"$(INTDIR)\graph.obj" \
	"$(INTDIR)\gslconversions.obj" \
	"$(INTDIR)\hclust.obj" \
	"$(INTDIR)\imputation.obj" \
	"$(INTDIR)\induce.obj" \
	"$(INTDIR)\jit_linker.obj" \
	"$(INTDIR)\knn.obj" \
	"$(INTDIR)\learn.obj" \
	"$(INTDIR)\lib_components.obj" \
	"$(INTDIR)\lib_io.obj" \
	"$(INTDIR)\lib_kernel.obj" \
	"$(INTDIR)\lib_learner.obj" \
	"$(INTDIR)\lib_preprocess.obj" \
	"$(INTDIR)\lib_vectors.obj" \
	"$(INTDIR)\linreg.obj" \
	"$(INTDIR)\logfit.obj" \
	"$(INTDIR)\logistic.obj" \
	"$(INTDIR)\logreg.obj" \
	"$(INTDIR)\lookup.obj" \
	"$(INTDIR)\lsq.obj" \
	"$(INTDIR)\lwr.obj" \
	"$(INTDIR)\majority.obj" \
	"$(INTDIR)\measures.obj" \
	"$(INTDIR)\meta.obj" \
	"$(INTDIR)\minimal_complexity.obj" \
	"$(INTDIR)\minimal_error.obj" \
	"$(INTDIR)\nearest.obj" \
	"$(INTDIR)\numeric_interface.obj" \
	"$(INTDIR)\orange.obj" \
	"$(INTDIR)\orvector.obj" \
	"$(INTDIR)\pnn.obj" \
	"$(INTDIR)\preprocessors.obj" \
	"$(INTDIR)\progress.obj" \
	"$(INTDIR)\pythonvars.obj" \
	"$(INTDIR)\r_imports.obj" \
	"$(INTDIR)\random.obj" \
	"$(INTDIR)\rconversions.obj" \
	"$(INTDIR)\readdata.obj" \
	"$(INTDIR)\redundancy.obj" \
	"$(INTDIR)\retisinter.obj" \
	"$(INTDIR)\root.obj" \
	"$(INTDIR)\rulelearner.obj" \
	"$(INTDIR)\spec_contingency.obj" \
	"$(INTDIR)\spec_gen.obj" \
	"$(INTDIR)\stringvars.obj" \
	"$(INTDIR)\subsets.obj" \
	"$(INTDIR)\survival.obj" \
	"$(INTDIR)\svm.obj" \
	"$(INTDIR)\symmatrix.obj" \
	"$(INTDIR)\tabdelim.obj" \
	"$(INTDIR)\table.obj" \
	"$(INTDIR)\tdidt.obj" \
	"$(INTDIR)\tdidt_split.obj" \
	"$(INTDIR)\tdidt_stop.obj" \
	"$(INTDIR)\transdomain.obj" \
	"$(INTDIR)\transval.obj" \
	"$(INTDIR)\trindex.obj" \
	"$(INTDIR)\valuelisttemplate.obj" \
	"$(INTDIR)\values.obj" \
	"$(INTDIR)\vars.obj" \
	"..\..\lib\include_d.lib"

"..\..\orange.pyd" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 


!IF "$(NO_EXTERNAL_DEPS)" != "1"
!IF EXISTS("Orange.dep")
!INCLUDE "Orange.dep"
!ELSE 
!MESSAGE Warning: cannot find "Orange.dep"
!ENDIF 
!ENDIF 


!IF "$(CFG)" == "Orange - Win32 Release" || "$(CFG)" == "Orange - Win32 Debug" || "$(CFG)" == "Orange - Win32 Release_Debug"
SOURCE=.\assistant.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\assistant.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\assistant.obj"	"$(INTDIR)\assistant.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\assistant.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\assoc.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\assoc.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\assoc.obj"	"$(INTDIR)\assoc.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\assoc.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\assoc_sparse.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\assoc_sparse.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\assoc_sparse.obj"	"$(INTDIR)\assoc_sparse.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\assoc_sparse.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\basket.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\basket.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\basket.obj"	"$(INTDIR)\basket.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\basket.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\basstat.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\basstat.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\basstat.obj"	"$(INTDIR)\basstat.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\basstat.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\bayes.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\bayes.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\bayes.obj"	"$(INTDIR)\bayes.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\bayes.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\boolcnt.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\boolcnt.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\boolcnt.obj"	"$(INTDIR)\boolcnt.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\boolcnt.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\c4.5.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\c4.5.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\c4.5.obj"	"$(INTDIR)\c4.5.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\c4.5.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\c45inter.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\c45inter.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\c45inter.obj"	"$(INTDIR)\c45inter.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\c45inter.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\calibrate.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\calibrate.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\calibrate.obj"	"$(INTDIR)\calibrate.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\calibrate.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\callback.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\callback.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\callback.obj"	"$(INTDIR)\callback.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\callback.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cartesian.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cartesian.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cartesian.obj"	"$(INTDIR)\cartesian.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cartesian.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\clas_gen.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\clas_gen.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\clas_gen.obj"	"$(INTDIR)\clas_gen.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\clas_gen.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\classfromvar.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\classfromvar.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\classfromvar.obj"	"$(INTDIR)\classfromvar.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\classfromvar.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\classifier.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\classifier.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\classifier.obj"	"$(INTDIR)\classifier.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\classifier.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cls_example.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cls_example.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cls_example.obj"	"$(INTDIR)\cls_example.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cls_example.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cls_misc.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cls_misc.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cls_misc.obj"	"$(INTDIR)\cls_misc.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cls_misc.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cls_orange.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cls_orange.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cls_orange.obj"	"$(INTDIR)\cls_orange.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cls_orange.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cls_value.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cls_value.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cls_value.obj"	"$(INTDIR)\cls_value.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cls_value.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\contingency.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\contingency.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\contingency.obj"	"$(INTDIR)\contingency.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\contingency.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\converts.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\converts.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\converts.obj"	"$(INTDIR)\converts.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\converts.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\cost.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\cost.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\cost.obj"	"$(INTDIR)\cost.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\cost.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\costwrapper.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\costwrapper.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\costwrapper.obj"	"$(INTDIR)\costwrapper.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\costwrapper.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\decomposition.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\decomposition.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\decomposition.obj"	"$(INTDIR)\decomposition.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\decomposition.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\dictproxy.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\dictproxy.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\dictproxy.obj"	"$(INTDIR)\dictproxy.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\dictproxy.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\discretize.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\discretize.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\discretize.obj"	"$(INTDIR)\discretize.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\discretize.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\dist_clustering.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\dist_clustering.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\dist_clustering.obj"	"$(INTDIR)\dist_clustering.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\dist_clustering.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\distance.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\distance.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\distance.obj"	"$(INTDIR)\distance.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\distance.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\distance_dtw.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\distance_dtw.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\distance_dtw.obj"	"$(INTDIR)\distance_dtw.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\distance_dtw.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\distancemap.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "px" /I "ppp" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /Gs /c 

"$(INTDIR)\distancemap.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /Zi /Od /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fr"$(INTDIR)\\" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\distancemap.obj"	"$(INTDIR)\distancemap.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /Fp"$(INTDIR)\Orange.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\distancemap.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\distvars.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\distvars.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\distvars.obj"	"$(INTDIR)\distvars.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\distvars.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\domain.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\domain.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\domain.obj"	"$(INTDIR)\domain.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\domain.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\domaindepot.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\domaindepot.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\domaindepot.obj"	"$(INTDIR)\domaindepot.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\domaindepot.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\errors.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\errors.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\errors.obj"	"$(INTDIR)\errors.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\errors.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\estimateprob.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\estimateprob.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\estimateprob.obj"	"$(INTDIR)\estimateprob.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\estimateprob.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\exampleclustering.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\exampleclustering.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\exampleclustering.obj"	"$(INTDIR)\exampleclustering.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\exampleclustering.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\examplegen.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\examplegen.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\examplegen.obj"	"$(INTDIR)\examplegen.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\examplegen.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\examples.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\examples.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\examples.obj"	"$(INTDIR)\examples.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\examples.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\excel.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\excel.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\excel.obj"	"$(INTDIR)\excel.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\excel.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\filegen.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\filegen.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\filegen.obj"	"$(INTDIR)\filegen.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\filegen.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\filter.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\filter.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\filter.obj"	"$(INTDIR)\filter.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\filter.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\functions.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\functions.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\functions.obj"	"$(INTDIR)\functions.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\functions.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\garbage.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\garbage.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\garbage.obj"	"$(INTDIR)\garbage.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\garbage.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\getarg.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\getarg.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\getarg.obj"	"$(INTDIR)\getarg.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\getarg.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\graph.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\graph.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\graph.obj"	"$(INTDIR)\graph.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\graph.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\gslconversions.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\gslconversions.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\gslconversions.obj"	"$(INTDIR)\gslconversions.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\gslconversions.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\hclust.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\hclust.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\hclust.obj"	"$(INTDIR)\hclust.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\hclust.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\imputation.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\imputation.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\imputation.obj"	"$(INTDIR)\imputation.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\imputation.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\induce.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\induce.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\induce.obj"	"$(INTDIR)\induce.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\induce.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\jit_linker.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\jit_linker.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\jit_linker.obj"	"$(INTDIR)\jit_linker.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\jit_linker.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\knn.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\knn.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\knn.obj"	"$(INTDIR)\knn.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\knn.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\learn.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\learn.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\learn.obj"	"$(INTDIR)\learn.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\learn.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_components.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_components.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_components.obj"	"$(INTDIR)\lib_components.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_components.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_io.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_io.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_io.obj"	"$(INTDIR)\lib_io.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_io.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_kernel.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_kernel.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_kernel.obj"	"$(INTDIR)\lib_kernel.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_kernel.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_learner.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_learner.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_learner.obj"	"$(INTDIR)\lib_learner.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_learner.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_preprocess.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_preprocess.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_preprocess.obj"	"$(INTDIR)\lib_preprocess.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_preprocess.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lib_vectors.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lib_vectors.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lib_vectors.obj"	"$(INTDIR)\lib_vectors.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lib_vectors.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\linreg.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\linreg.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\linreg.obj"	"$(INTDIR)\linreg.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\linreg.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\logfit.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\logfit.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\logfit.obj"	"$(INTDIR)\logfit.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\logfit.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\logistic.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\logistic.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\logistic.obj"	"$(INTDIR)\logistic.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\logistic.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\logreg.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\logreg.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\logreg.obj"	"$(INTDIR)\logreg.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\logreg.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lookup.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lookup.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lookup.obj"	"$(INTDIR)\lookup.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lookup.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lsq.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lsq.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lsq.obj"	"$(INTDIR)\lsq.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lsq.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\lwr.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\lwr.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\lwr.obj"	"$(INTDIR)\lwr.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\lwr.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\majority.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\majority.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\majority.obj"	"$(INTDIR)\majority.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\majority.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\measures.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\measures.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\measures.obj"	"$(INTDIR)\measures.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\measures.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\meta.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\meta.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\meta.obj"	"$(INTDIR)\meta.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\meta.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\minimal_complexity.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\minimal_complexity.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\minimal_complexity.obj"	"$(INTDIR)\minimal_complexity.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\minimal_complexity.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\minimal_error.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\minimal_error.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\minimal_error.obj"	"$(INTDIR)\minimal_error.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\minimal_error.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\nearest.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\nearest.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\nearest.obj"	"$(INTDIR)\nearest.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\nearest.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\numeric_interface.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\numeric_interface.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\numeric_interface.obj"	"$(INTDIR)\numeric_interface.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\numeric_interface.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\orange.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\orange.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\orange.obj"	"$(INTDIR)\orange.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\orange.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\orvector.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\orvector.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\orvector.obj"	"$(INTDIR)\orvector.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\orvector.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\pnn.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\pnn.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\pnn.obj"	"$(INTDIR)\pnn.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\pnn.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\preprocessors.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\preprocessors.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\preprocessors.obj"	"$(INTDIR)\preprocessors.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\preprocessors.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\progress.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\progress.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\progress.obj"	"$(INTDIR)\progress.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\progress.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\pythonvars.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\pythonvars.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\pythonvars.obj"	"$(INTDIR)\pythonvars.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\pythonvars.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\r_imports.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\r_imports.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\r_imports.obj"	"$(INTDIR)\r_imports.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\r_imports.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\random.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\random.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\random.obj"	"$(INTDIR)\random.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\random.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\rconversions.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\rconversions.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\rconversions.obj"	"$(INTDIR)\rconversions.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\rconversions.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\readdata.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\readdata.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\readdata.obj"	"$(INTDIR)\readdata.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\readdata.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\redundancy.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "px" /I "ppp" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /Gs /c 

"$(INTDIR)\redundancy.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /Zi /Od /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fr"$(INTDIR)\\" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\redundancy.obj"	"$(INTDIR)\redundancy.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /Fp"$(INTDIR)\Orange.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\redundancy.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\retisinter.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\retisinter.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\retisinter.obj"	"$(INTDIR)\retisinter.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\retisinter.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\root.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\root.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\root.obj"	"$(INTDIR)\root.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\root.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\rulelearner.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\rulelearner.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\rulelearner.obj"	"$(INTDIR)\rulelearner.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\rulelearner.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\spec_contingency.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\spec_contingency.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\spec_contingency.obj"	"$(INTDIR)\spec_contingency.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\spec_contingency.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\spec_gen.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\spec_gen.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\spec_gen.obj"	"$(INTDIR)\spec_gen.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\spec_gen.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\stringvars.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\stringvars.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\stringvars.obj"	"$(INTDIR)\stringvars.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\stringvars.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\subsets.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\subsets.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\subsets.obj"	"$(INTDIR)\subsets.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\subsets.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\survival.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\survival.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\survival.obj"	"$(INTDIR)\survival.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\survival.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\svm.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\svm.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\svm.obj"	"$(INTDIR)\svm.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\svm.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\symmatrix.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\symmatrix.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\symmatrix.obj"	"$(INTDIR)\symmatrix.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\symmatrix.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\tabdelim.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "$(PYTHON)\include" /I "px" /I "ppp" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /Gs /c 

"$(INTDIR)\tabdelim.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

CPP_SWITCHES=/nologo /MDd /W3 /Gm /GR /GX /Zi /Od /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /Fr"$(INTDIR)\\" /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /GZ /Zm700 /c 

"$(INTDIR)\tabdelim.obj"	"$(INTDIR)\tabdelim.sbr" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

CPP_SWITCHES=/nologo /MD /W3 /GR /GX /O2 /I "../include" /I "ppp" /I "px" /I "$(PYTHON)\include" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "ORANGE_EXPORTS" /D "LINK_C45" /Fp"$(INTDIR)\Orange.pch" /YX /Fo"$(INTDIR)\\" /Fd"$(INTDIR)\\" /FD /Zm700 /c 

"$(INTDIR)\tabdelim.obj" : $(SOURCE) "$(INTDIR)"
	$(CPP) @<<
  $(CPP_SWITCHES) $(SOURCE)
<<


!ENDIF 

SOURCE=.\table.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\table.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\table.obj"	"$(INTDIR)\table.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\table.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\tdidt.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\tdidt.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\tdidt.obj"	"$(INTDIR)\tdidt.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\tdidt.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\tdidt_split.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\tdidt_split.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\tdidt_split.obj"	"$(INTDIR)\tdidt_split.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\tdidt_split.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\tdidt_stop.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\tdidt_stop.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\tdidt_stop.obj"	"$(INTDIR)\tdidt_stop.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\tdidt_stop.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\transdomain.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\transdomain.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\transdomain.obj"	"$(INTDIR)\transdomain.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\transdomain.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\transval.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\transval.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\transval.obj"	"$(INTDIR)\transval.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\transval.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\trindex.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\trindex.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\trindex.obj"	"$(INTDIR)\trindex.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\trindex.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\valuelisttemplate.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\valuelisttemplate.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\valuelisttemplate.obj"	"$(INTDIR)\valuelisttemplate.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\valuelisttemplate.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\values.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\values.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\values.obj"	"$(INTDIR)\values.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\values.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

SOURCE=.\vars.cpp

!IF  "$(CFG)" == "Orange - Win32 Release"


"$(INTDIR)\vars.obj" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"


"$(INTDIR)\vars.obj"	"$(INTDIR)\vars.sbr" : $(SOURCE) "$(INTDIR)"


!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"


"$(INTDIR)\vars.obj" : $(SOURCE) "$(INTDIR)"


!ENDIF 

!IF  "$(CFG)" == "Orange - Win32 Release"

"include - Win32 Release" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" 
   cd "..\orange"

"include - Win32 ReleaseCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release" RECURSE=1 CLEAN 
   cd "..\orange"

!ELSEIF  "$(CFG)" == "Orange - Win32 Debug"

"include - Win32 Debug" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" 
   cd "..\orange"

"include - Win32 DebugCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Debug" RECURSE=1 CLEAN 
   cd "..\orange"

!ELSEIF  "$(CFG)" == "Orange - Win32 Release_Debug"

"include - Win32 Release_Debug" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release_Debug" 
   cd "..\orange"

"include - Win32 Release_DebugCLEAN" : 
   cd "\D\ai\Orange\source\include"
   $(MAKE) /$(MAKEFLAGS) /F .\include.mak CFG="include - Win32 Release_Debug" RECURSE=1 CLEAN 
   cd "..\orange"

!ENDIF 


!ENDIF 

