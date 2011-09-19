all:	objdir $(OLD)/orange.so

MODULENAME=ORANGE
include ../makefile.defs
-include makefile.deps


#########################################################################
# If BLAS_LIB is not defiend then include a minimal subset of blas into
# orange library. If BLAS_LIB is defined it must be the name of the blas
# library to link with (e.g. blas). The libraries must be on the
# search path. 
#########################################################################

ifndef BLAS_LIB
BLAS_OBJECTS = obj/daxpy.o obj/ddot.o obj/dnrm2.o obj/dscal.o obj/dcopy.o
else
BLAS_OBJECTS =
LINKOPTIONS += -l$(BLAS_LIB)
endif

obj/%.o : blas/%.c blas/blas.h blas/blasp.h
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@


#########################################################################
# If LIBLINEAR_LIB is not defiend then include a minimal subset of LIBLINEAR
# into orange library. If LIBLINEAR_LIB is defined it must be the name of the
# liblinear library to link with (e.g. linear). The library must be on the
# search path. 
######################################################################### 

ifndef LIBLINEAR_LIB
LIBLINEAR_OBJECTS = obj/linear.o obj/tron.o
COMPILEOPTIONS += -Iliblinear # Add liblinear inclue dir to compile options
else
LIBLINEAR_OBJECTS =
LINKOPTIONS += -l$(LIBLINEAR_LIB)
endif

obj/%.o : liblinear/%.cpp liblinear/%.h
	$(COMPILER) $(COMPILEOPTIONS) -c $< -o $@


#########################################################################
# If LIBSVM_LIB is not defiend then include a minimal subset of LIBSVM
# into orange library. If LIBSVM_LIB is defined it must be the name of the
# libsvm library to link with (e.g. svm). The library must be on the 
# search path. 
#########################################################################

ifndef LIBSVM_LIB
LIBSVM_OBJECTS = obj/svm.o
else
LIBSVM_OBJECTS =
LINKOPTIONS += -l$(LIBSVM_LIB)
endif

obj/%.o : libsvm/%.cpp libsvm/%.h
	$(COMPILER) $(COMPILEOPTIONS) -c $< -o $@
	

LINPACK_OBJECTS = obj/dqrsl.o obj/dqrdc2.o obj/dtrsl.o obj/linpack.o

obj/%.o : linpack/%.c linpack/linpack.h
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@

ALL_OBJECTS = $(ORANGE_OBJECTS) $(BLAS_OBJECTS) $(LINPACK_OBJECTS) $(LIBLINEAR_OBJECTS) $(LIBSVM_OBJECTS)

$(OLD)/orange.so:	ppp/stamp px/stamp $(ALL_OBJECTS)
	$(LINKER) $(ALL_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orange.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orange.so $(OLD)/orange.so
endif
	cd $(OLD); ln -sf orange.so $(LIBRARY_FILE)

clean: cleantemp
#	rm -f r_imports.ipp r_imports.jpp
	rm -f $(OLD)/orange.so $(OLD)/liborange.so
