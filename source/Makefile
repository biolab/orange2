############
# Definitions of directories

ORANGEHOME= ..

SOURCEDIR = $(ORANGEHOME)/source
ODIR = /tmp/orange
BINDIR = $(ORANGEHOME)/modules

ORANGEDIR  = $(SOURCEDIR)/orange
INCLUDEDIR = $(SOURCEDIR)/include
CORNDIR = $(SOURCEDIR)/corn
STATCDIR = $(SOURCEDIR)/statc

DEVSCRIPTS= $(ORANGEDIR)/devscripts
PPPDIR = $(ORANGEDIR)/ppp
PXDIR = $(ORANGEDIR)/px

EXTERNALDIR = $(ORANGEHOME)/source/external

# Lines below determine the directory with Python.h.
# If this does not work, find Python.h yourself and enter
# the directory below.
PYTHONINCLUDE = $(shell python -c "import sys; print sys.prefix+'/include/python'+sys.version[:3]")
PYTHONLIB = $(shell python -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")

LOGREGD = $(EXTERNALDIR)/logreg
LOGREG_OBJECTS = $(LOGREGD)/logreg.cpp $(LOGREGD)/lsq.cpp

ORANGE_SO = $(BINDIR)/orange.so $(BINDIR)/corn.so $(BINDIR)/statc.so

INCLUDEPATH = -I$(PPPDIR) -I$(PXDIR) -I$(ORANGEDIR) \
		-I$(INCLUDEDIR) -I$(CORNDIR) -I$(STATCDIR) \
		-I$(EXTERNALDIR) -I$(PYTHONINCLUDE)

LIBPATH = -L$(LOGREGD)

############
# Paths for extensions

vpath %.cpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.hpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.h $(INCLUDEDIR):$(PYTHONINCLUDE)
vpath %.ppp $(PPPDIR)
vpath %.px $(PXDIR)


############
# Compiler options

CC=	gcc
#CXX=	g++ $(INCLUDEPATH) -fPIC -Wno-deprecated -Wno-sign-compare -Wall -ggdb3
CXX=	g++ $(INCLUDEPATH) -fPIC -w -DLINUX -O3
LINKCC=	$(PURIFY) $(CC)

LNKSTL=	-lstdc++



############
#   .ppp and .px files are created by pyprops and pyxtract
#   which get called to build timestamp.h. The reason is that
#   all .ppp/.px files are created at the same time, in a single
#   run of the corresponding tool; the timestamp's time is compared
#   to a list of cpp/hpp files.

$(ODIR)/%.o :
	$(CXX) -c $< -o $@


all	: dirs $(ORANGE_SO)

include makefile.deps

dirs:
	-@mkdir -p $(ODIR) $(BINDIR) > /dev/null

$(BINDIR)/orange.so: $(PPPDIR)/stamp $(PXDIR)/stamp $(ORANGE_OBJECTS) $(LOGREG_OBJECTS)
	gcc $(ORANGE_OBJECTS) $(LOGREG_OBJECTS) $(ODIR)/logreg.a -o $(BINDIR)/orange.so -shared $(LNKSTL)

$(BINDIR)/corn.so:	$(CORN_OBJECTS)
	gcc $(CORN_OBJECTS) -o $(BINDIR)/corn.so -shared $(LNKSTL)

$(BINDIR)/statc.so:	$(STATC_OBJECTS)
	gcc $(STATC_OBJECTS) -o $(BINDIR)/statc.so -shared $(LNKSTL)


install:	all
	-mkdir -p $(PYTHONLIB)
	$(warning If this fails, re-run 'make install' as root, or just set 'export PYTHONPATH=$(subst source,modules,$(shell pwd))')
	mv $(BINDIR)/*.so $(PYTHONLIB)

dist:
	cd .. ; \
        sh ../dedos ; \
	tar -czf orange09.tgz source doc modules copying building \
		--exclude *.ncb --exclude *.plg --exclude *.ppp --exclude *.px --exclude *.bak --exclude *.opt --exclude stamp --exclude cvs --exclude *.so --exclude *.pyc --exclude *.pyd --exclude *.dll \
		--exclude CVS --exclude outputs \
		--exclude OrangeCanvas --exclude GUIApplications --exclude OrangeWidgets --exclude external/*.c --exclude external/*.i ; \
	cd source


clean:
	rm -f $(ODIR)/* $(BINDIR)/*.so $(PPPDIR)/* $(PXDIR)/*
