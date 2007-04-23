all:	objdir $(OLD)/orange.so

MODULENAME=ORANGE
include ../makefile.defs
-include makefile.deps

r_imports.jpp: r_imports.hpp
	python ../pyxtract/jitlink_build.py R.so r_imports.hpp
r_imports.ipp: r_imports.hpp
	python ../pyxtract/jitlink_build.py R.so r_imports.hpp

$(OLD)/orange.so:	ppp/stamp px/stamp $(ORANGE_OBJECTS)
	$(LINKER) $(ORANGE_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orange.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(OLD)/orange.so $(OLD)/orange.so
endif
	cd $(OLD); pwd; ln -sf orange.so $(LIBRARY_FILE)

clean:	cleantemp
	rm -f r_imports.ipp r_imports.jpp
	rm -f $(OLD)/orange.so $(OLD)/liborange.so $(OLD)/liborange.dylib
