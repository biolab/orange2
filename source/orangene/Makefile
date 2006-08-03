all:	objdir $(OLD)/orangene.so

MODULENAME=ORANGENE
include ../makefile.defs
include makefile.deps

$(OLD)/orangene.so:	ppp/stamp px/stamp $(ORANGENE_OBJECTS)
	$(CCL) $(ORANGENE_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangene.so

clean:	cleantemp
	rm -f $(OLD)/orangene.so
