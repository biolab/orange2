all:	objdir $(OLD)/orangemds.so

MODULENAME=ORANGEMDS
include ../makefile.defs
include makefile.deps

$(OLD)/orangemds.so: px/stamp $(ORANGEMDS_OBJECTS)
	gcc $(ORANGEMDS_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangemds.so

clean:	cleantemp
	rm -f $(OLD)/orangemds.so
