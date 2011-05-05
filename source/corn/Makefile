all:	objdir $(OLD)/corn.so

MODULENAME = CORN
include ../makefile.defs

obj/corn.o:	corn.cpp corn.hpp

$(OLD)/corn.so:	obj/corn.o
	$(LINKER) obj/corn.o $(LINKOPTIONS) -o $(OLD)/corn.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/corn.so $(OLD)/corn.so
endif

clean:	cleantemp
	rm -f $(OLD)/corn.so
