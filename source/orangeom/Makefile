all:	objdir $(OLD)/orangeom.so

MODULENAME=ORANGEOM
include ../makefile.defs
include makefile.deps

obj/WmlDelaunay2a.o : wml/WmlDelaunay2a.cpp
obj/WmlMath.o : wml/WmlMath.cpp
obj/WmlSystem.o : wml/WmlSystem.cpp
obj/WmlVector2.o : wml/WmlVector2.cpp

$(WML_OBJECTS):	obj/WmlDelaunay2a.o obj/WmlMath.o obj/WmlSystem.o obj/WmlVector2.o

$(OLD)/orangeom.so:	px/stamp $(ORANGEOM_OBJECTS)
	gcc $(ORANGEOM_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so

clean:	cleantemp
	rm -f $(OLD)/orangeom.so
