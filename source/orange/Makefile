all:	objdir $(OLD)/orange.so

MODULENAME=ORANGE
include ../makefile.defs
include makefile.deps

r_imports.ipp rimporst.jpp:	r_imporst.hpp
	python ../pyxtract/jitlink_build.py R.so r_imports.hpp

$(OLD)/orange.so:	ppp/stamp px/stamp $(ORANGE_OBJECTS)
	gcc $(ORANGE_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orange.so -lgsl -lgslcblas
	cd $(OLD); pwd; ln -sf orange.so liborange.so

clean:	cleantemp
	rm -f $(OLD)/orange.so $(OLD)/liborange.so
