all:	objdir ../orangeom.so

MODULENAME = ORANGEOM
include ../makefile.defs
include makefile.deps

objdir:
	-@mkdir -p obj > /dev/null

../orangeom.so:	ppp/stamp px/stamp $(ORANGEOM_OBJECTS)
	gcc $(ORANGEOM_OBJECTS) $(LINKOPTIONS) -o ../orangeom.so

