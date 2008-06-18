all:	objdir $(OLD)/statc.so

MODULENAME = STATC
include ../makefile.defs

obj/statc.o:	statc.cpp statc.hpp \
		../include/stat.hpp ../include/stat.cpp $(OLD)/$(LIBRARY_FILE)

$(OLD)/statc.so:	obj/statc.o
	$(LINKER) obj/statc.o $(LINKOPTIONS) -o $(OLD)/statc.so

clean:	cleantemp
	rm -f $(OLD)/statc.so
