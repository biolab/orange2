############
# Definitions of directories

ORANGEHOME= ..

SOURCEDIR = $(ORANGEHOME)/source
ODIR = $(ORANGEHOME)/o
BINDIR = $(ORANGEHOME)/bin

ORANGEDIR  = $(SOURCEDIR)/orange
INCLUDEDIR = $(SOURCEDIR)/include
CORNDIR = $(SOURCEDIR)/corn
STATCDIR = $(SOURCEDIR)/statc

DEVSCRIPTS= $(ORANGEDIR)/devscripts
PPPDIR = $(ORANGEDIR)/ppp
PXDIR = $(ORANGEDIR)/px

EXTERNALDIR = $(ORANGEHOME)/external

# Below is used for determining directory with Python.h.
# If this does not work, find Python.h yourself and enter
# the directory below.
PYTHONINCLUDE = $(shell python -c "import sys; print sys.prefix+'/include/python'+sys.version[:3]")
PYTHONLIB = $(shell python -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")

C45D = $(EXTERNALDIR)/c45
C45_OBJECTS = $(C45D)/besttree.o $(C45D)/build.o $(C45D)/classify.o $(C45D)/confmat.o \
		$(C45D)/contin.o $(C45D)/discr.o $(C45D)/genlogs.o $(C45D)/getdata.o \
		$(C45D)/getnames.o $(C45D)/getopt.o $(C45D)/header.o $(C45D)/info.o \
		$(C45D)/prune.o $(C45D)/sort.o $(C45D)/st-thresh.o $(C45D)/stats.o \
		$(C45D)/subset.o $(C45D)/trees.o

ORANGE_SO = $(BINDIR)/orange.so $(BINDIR)/corn.so $(BINDIR)/statc.so


INCLUDEPATH = -I$(PPPDIR) -I$(PXDIR) -I$(ORANGEDIR) \
		-I$(INCLUDEDIR) -I$(CORNDIR) -I$(STATCDIR) \
		-I$(EXTERNALDIR) -I$(PYTHONINCLUDE)

LIBPATH = -L$(EXTERNALDIR)/c45

############
# Paths for extensions

vpath %.cpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.hpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.h $(INCLUDEDIR):$(PYTHONINCLUDE)
vpath %.i $(C45D)
vpath %.ppp $(PPPDIR)
vpath %.px $(PXDIR)


############
# Compiler options

CC=	gcc
#CXX=	g++ $(INCLUDEPATH) -fPIC -Wno-deprecated -Wno-sign-compare -Wall -DLINK_C45 -ggdb3
#CXX=	g++ $(INCLUDEPATH) -fPIC -Wno-deprecated -Wno-sign-compare -Wall -DLINK_C45 -O3
CXX=	g++ $(INCLUDEPATH) -fPIC -w -DLINK_C45 -O3
LINKCC=	$(PURIFY) $(CC)

LNKSTL=	-lstdc++



############
# Patterns:
#   .ppp and .px files are created by pyprops and pyxtract
#   which get called to build timestamp.h. The reason is that
#   all .ppp/.px files are created at the same time, in a single
#   run of the corresponding tool; the timestamp's time is compared
#   to a list of cpp/hpp files.

#%.ppp:	
#	
#%.px:
#	
$(ODIR)/%.o :
	$(CXX) -c $< -o $@

############
# Rules

all	: dirs $(ORANGE_SO)

include makefile.deps

dirs:
	-@mkdir -p $(ODIR) $(BINDIR) > /dev/null

$(BINDIR)/orange.so: $(PPPDIR)/stamp $(PXDIR)/stamp $(ODIR)/c45.a $(ORANGE_OBJECTS)
	gcc $(ORANGE_OBJECTS) $(ODIR)/c45.a -o $(BINDIR)/orange.so -shared $(LNKSTL)

$(BINDIR)/corn.so:	$(CORN_OBJECTS)
	gcc $(CORN_OBJECTS) -o $(BINDIR)/corn.so -shared $(LNKSTL)

$(BINDIR)/statc.so:	$(STATC_OBJECTS)
	gcc $(STATC_OBJECTS) -o $(BINDIR)/statc.so -shared $(LNKSTL)

$(ODIR)/c45.a:	$(C45_OBJECTS)
	ar r $(ODIR)/c45.a $(C45_OBJECTS)


install:	all
	-mkdir -p $(PYTHONLIB)
	$(warning If this fails, re-run 'make install' as root, or just set 'export PYTHONPATH=$(subst source,bin,$(shell pwd))')
	mv $(BINDIR)/*.so $(PYTHONLIB)

dist:
	cd .. ; \
	tar -czf orange07.tgz source copying building \
		external/c45/rand_free.c external/c45/C45.dsp external/c45/directory_for_c45_files \
		--exclude *.ncb --exclude *.plg --exclude *.ppp --exclude *.px --exclude *.bak --exclude *.opt --exclude stamp


clean:
	rm -f $(ODIR)/* $(BINDIR)/* $(PPPDIR)/* $(PXDIR)/*
