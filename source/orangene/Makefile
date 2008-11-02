all:	objdir $(OLD)/orangene.so

MODULENAME=ORANGENE
include ../makefile.defs
-include makefile.deps

$(OLD)/orangene.so:	ppp/stamp px/stamp $(ORANGENE_OBJECTS)
	$(LINKER) $(ORANGENE_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangene.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orangene.so $(OLD)/orangene.so
endif

clean:	cleantemp
	rm -f $(OLD)/orangene.so
