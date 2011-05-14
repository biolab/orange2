all:	objdir $(OLD)/orange.so

MODULENAME=ORANGE
include ../makefile.defs
-include makefile.deps


r_imports.jpp: r_imports.hpp
	$(PYTHON) ../pyxtract/jitlink_build.py R.so r_imports.hpp
r_imports.ipp: r_imports.hpp
	$(PYTHON) ../pyxtract/jitlink_build.py R.so r_imports.hpp

obj/daxpy.o : blas/daxpy.c 
	$(CCOMPILER) $(COMPILEOPTIONS) -c blas/daxpy.c  -o obj/daxpy.o
	
obj/ddot.o : blas/ddot.c 
	$(CCOMPILER) $(COMPILEOPTIONS) -c blas/ddot.c  -o obj/ddot.o

obj/dnrm2.o : blas/dnrm2.c 
	$(CCOMPILER) $(COMPILEOPTIONS) -c blas/dnrm2.c  -o obj/dnrm2.o

obj/dscal.o : blas/dscal.c 
	$(CCOMPILER) $(COMPILEOPTIONS) -c blas/dscal.c  -o obj/dscal.o

BLAS_OBJECTS = obj/daxpy.o obj/ddot.o obj/dnrm2.o obj/dscal.o

$(OLD)/orange.so:	ppp/stamp px/stamp $(ORANGE_OBJECTS) $(BLAS_OBJECTS)
	$(LINKER) $(ORANGE_OBJECTS) $(BLAS_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orange.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orange.so $(OLD)/orange.so
endif
	cd $(OLD); ln -sf orange.so $(LIBRARY_FILE)

clean: cleantemp
	rm -f r_imports.ipp r_imports.jpp
	rm -f $(OLD)/orange.so $(OLD)/liborange.so
