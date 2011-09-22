all:	objdir $(OLD)/orangeom.so

MODULENAME = ORANGEOM
include ../makefile.defs
-include makefile.deps


#########################################################################
# If QHULL_LIB is not defiend then include qhull into orange library.
# If QHULL_LIB is defined it must be the name of the qhull library to
# link with (e.g. qhull). The library and its header files must be on the
# search path (use CFLAGS/LDFLAGS if not on the default path).
######################################################################### 

ifndef QHULL_LIB
QHULL_OBJECTS = obj/qhull_geom.o obj/qhull_geom2.o obj/qhull_global.o obj/qhull_io.o\
		obj/qhull_mem.o obj/qhull_merge.o obj/qhull_poly.o obj/qhull_poly2.o obj/qhull_qhull.o\
		obj/qhull_qset.o obj/qhull_stat.o obj/qhull_user.o
# Add curr dir to include path
COMPILEOPTIONS += -I./
else
QHULL_OBJECTS = 
LINKOPTIONS += -l$(QHULL_LIB)
endif

obj/qhull_%.o : qhull/%.c
	$(CCOMPILER) $(COMPILEOPTIONS) -c $< -o $@

$(OLD)/orangeom.so:	px/stamp ppp/stamp $(ORANGEOM_OBJECTS) $(QHULL_OBJECTS)
	$(LINKER) $(ORANGEOM_OBJECTS) $(QHULL_OBJECTS) $(LINKOPTIONS) -o $(OLD)/orangeom.so
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orangeom.so $(OLD)/orangeom.so
endif

clean:	cleantemp
	rm -f lib_vectors.cpp lib_vectors_auto.txt
	rm -f $(OLD)/orangeom.so
