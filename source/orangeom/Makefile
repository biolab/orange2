all:	objdir $(OLD)/orangeom.so

MODULENAME=ORANGEOM
include ../makefile.defs
include makefile.deps

obj/WmlDelaunay2a.o : wml/WmlDelaunay2a.cpp
obj/WmlMath.o : wml/WmlMath.cpp
obj/WmlSystem.o : wml/WmlSystem.cpp
obj/WmlVector2.o : wml/WmlVector2.cpp

obj/som_pak_DATA.o : som/datafile.c
obj/som_pak_LVQ.o : som/lvq_pak.c
obj/som_pak_SOM.o : som/som_rout.c
obj/som_pak_FILE.o : som/fileio.c
obj/som_pak_VERSION.o : som/version.c
obj/som_pak_LABELS.o : som/labels.c

WML_OBJECTS=obj/WmlDelaunay2a.o obj/WmlMath.o obj/WmlSystem.o obj/WmlVector2.o
SOM_OBJECTS=obj/som_pak_DATA.o obj/som_pak_LVQ.o obj/som_pak_SOM.o obj/som_pak_FILE.o obj/som_pak_VERSION.o obj/som_pak_LABELS.o

$(OLD)/orangeom.so:	px/stamp ppp/stamp $(ORANGEOM_OBJECTS) $(WML_OBJECTS) $(SOM_OBJECTS)
	gcc $(ORANGEOM_OBJECTS) $(WML_OBJECTS) $(SOM_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so

clean:	cleantemp
	rm -f $(OLD)/orangeom.so
