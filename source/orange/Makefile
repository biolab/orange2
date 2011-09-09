all:	objdir $(OLD)/orange.so

MODULENAME=ORANGE
include ../makefile.defs
-include makefile.deps


r_imports.jpp: r_imports.hpp
	$(PYTHON) ../pyxtract/jitlink_build.py R.so r_imports.hpp
r_imports.ipp: r_imports.hpp
	$(PYTHON) ../pyxtract/jitlink_build.py R.so r_imports.hpp


# TODO: Dont compile BLAS if we can link with system (or user
# supplied) blas

BLAS_OBJECTS = obj/daxpy.o obj/ddot.o obj/dnrm2.o obj/dscal.o obj/dcopy.o
	
obj/%.o : blas/%.c blas/blas.h blas/blasp.h
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@

LINPACK_OBJECTS = obj/dqrsl.o obj/dqrdc2.o obj/dtrsl.o obj/linpack.o

obj/%.o : linpack/%.c linpack/linpack.h
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@

$(OLD)/orange.so:	ppp/stamp px/stamp $(ORANGE_OBJECTS) $(BLAS_OBJECTS) $(LINPACK_OBJECTS)
	$(LINKER) $(ORANGE_OBJECTS) $(BLAS_OBJECTS) $(LINPACK_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orange.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orange.so $(OLD)/orange.so
endif
	cd $(OLD); ln -sf orange.so $(LIBRARY_FILE)

clean: cleantemp
	rm -f r_imports.ipp r_imports.jpp
	rm -f $(OLD)/orange.so $(OLD)/liborange.so
