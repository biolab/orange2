all:	objdir $(OLD)/orangeom.so

MODULENAME=ORANGEOM
include ../makefile.defs
include makefile.deps

$(OLD)/orangeom.so:	px/stamp $(ORANGEOM_OBJECTS)
	gcc $(ORANGEOM_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so

clean:	cleantemp
	rm -f $(OLD)/orangeom.so
