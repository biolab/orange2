all:	objdir $(OLD)/statc.so

MODULENAME = STATC
include ../makefile.defs

obj/statc.o:	statc.cpp statc.hpp \
		../include/stat.hpp ../include/stat.cpp $(OLD)/$(LIBRARY_FILE)

$(OLD)/statc.so:	obj/statc.o
	$(LINKER) obj/statc.o $(LINKOPTIONS) -o $(OLD)/statc.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/statc.so $(OLD)/statc.so
endif

clean:	cleantemp
	rm -f $(OLD)/statc.so
