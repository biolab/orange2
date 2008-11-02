all:	objdir $(OLD)/orangeom.so

MODULENAME=ORANGEOM
include ../makefile.defs
-include makefile.deps

obj/qhull_geom.o : qhull/geom.c
obj/qhull_geom2.o : qhull/geom2.c
obj/qhull_global.o : qhull/global.c
obj/qhull_io.o : qhull/io.c
obj/qhull_mem.o : qhull/mem.c
obj/qhull_merge.o : qhull/merge.c
obj/qhull_poly.o : qhull/poly.c
obj/qhull_poly2.o : qhull/poly2.c
obj/qhull_qhull.o : qhull/qhull.c
obj/qhull_qset.o : qhull/qset.c
obj/qhull_stat.o : qhull/stat.c
obj/qhull_user.o : qhull/user.c

obj/qhull_%.o : qhull/%.c
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@

QHULL_OBJECTS=obj/qhull_geom.o obj/qhull_geom2.o obj/qhull_global.o obj/qhull_io.o obj/qhull_mem.o obj/qhull_merge.o obj/qhull_poly.o obj/qhull_poly2.o obj/qhull_qhull.o obj/qhull_qset.o obj/qhull_stat.o obj/qhull_user.o

$(OLD)/orangeom.so:	px/stamp ppp/stamp $(ORANGEOM_OBJECTS) $(QHULL_OBJECTS)
	$(LINKER) $(ORANGEOM_OBJECTS) $(QHULL_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orangeom.so $(OLD)/orangeom.so
endif

clean:	cleantemp
	rm -f lib_vectors.cpp lib_vectors_auto.txt
	rm -f $(OLD)/orangeom.so
