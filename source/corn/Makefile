all:	objdir $(OLD)/corn.so

MODULENAME = CORN
include ../makefile.defs

obj/corn.o:	corn.cpp corn.hpp
$(OLD)/corn.so:	corn.o
	gcc corn.o $(LINKOPTIONS) -o $(OLD)/corn.so

clean:	cleantemp
	rm -f $(OLD)/corn.so
