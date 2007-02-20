all:	objdir $(OLD)/orangeom.so

MODULENAME=ORANGEOM
include ../makefile.defs
include makefile.deps

obj/som_pak_DATA.o : som/datafile.c
obj/som_pak_LVQ.o : som/lvq_pak.c
obj/som_pak_SOM.o : som/som_rout.c
obj/som_pak_FILE.o : som/fileio.c
obj/som_pak_VERSION.o : som/version.c
obj/som_pak_LABELS.o : som/labels.c

obj/geom.o : qhull/geom.c
obj/geom2.o : qhull/geom2.c
obj/global.o : global.c
obj/io.o : qhull/io.c
obj/mem.o : qhull/mem.c
obj/merge.o : qhull/merge.c
obj/poly.o : qhull/poly.c
obj/poly2.o : qhull/poly2.c
obj/qhull.o : qhull/qhull.c
obj/qset.o : qhull/qset.c
obj/stat.o : qhull/stat.c
obj/user.o : qhull/user.c


SOM_OBJECTS=obj/som_pak_DATA.o obj/som_pak_LVQ.o obj/som_pak_SOM.o obj/som_pak_FILE.o obj/som_pak_VERSION.o obj/som_pak_LABELS.o
QHULL_OBJECTS=obj/geom.o obj/geom2.o obj/global.o obj/io.o obj/mem.o obj/merge.o obj/poly.o obj/poly2.o obj/qhull.o obj/qset.o obj/stat.o obj/user.o

$(OLD)/orangeom.so:	px/stamp ppp/stamp $(ORANGEOM_OBJECTS) $(SOM_OBJECTS) $(QHULL_OBJECTS)
	$(LINKER) $(ORANGEOM_OBJECTS) $(SOM_OBJECTS) $(QHULL_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so

clean:	cleantemp
	rm -f $(OLD)/orangeom.so
