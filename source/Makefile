############
# Definitions of directories

ORANGEHOME= ..

SOURCEDIR = $(ORANGEHOME)/source
ODIR = $(ORANGEHOME)/obj
BINDIR = $(ORANGEHOME)/modules

ORANGEDIR  = $(SOURCEDIR)/orange
INCLUDEDIR = $(SOURCEDIR)/include
CORNDIR = $(SOURCEDIR)/corn
STATCDIR = $(SOURCEDIR)/statc

DEVSCRIPTS= $(ORANGEDIR)/devscripts
PPPDIR = $(ORANGEDIR)/ppp
PXDIR = $(ORANGEDIR)/px

# Lines below determine the directory with Python.h.
# If this does not work, find Python.h yourself and enter
# the directory below.
PYTHONINCLUDE = $(shell python -c "import sys; print sys.prefix+'/include/python'+sys.version[:3]")
PYTHONLIB = $(shell python -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")

ORANGE_SO = $(BINDIR)/orange.so $(BINDIR)/corn.so $(BINDIR)/statc.so

INCLUDEPATH = -I$(PPPDIR) -I$(PXDIR) -I$(ORANGEDIR) \
		-I$(INCLUDEDIR) -I$(CORNDIR) -I$(STATCDIR) \
		-I$(PYTHONINCLUDE) -I$(LOGREGD)

LIBPATH = -L$(LOGREGD)

############
# Paths for extensions

vpath %.cpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.hpp $(ORANGEDIR):$(CORNDIR):$(STATCDIR):$(INCLUDEDIR)
vpath %.h $(ORANGEDIR):$(INCLUDEDIR):$(PYTHONINCLUDE)
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

$(BINDIR)/orange.so: $(PPPDIR)/stamp $(PXDIR)/stamp $(ORANGE_OBJECTS)
	gcc $(ORANGE_OBJECTS) -fPIC -o $(BINDIR)/orange.so -shared $(LNKSTL)

$(BINDIR)/corn.so:	$(CORN_OBJECTS)
	gcc $(CORN_OBJECTS) -o $(BINDIR)/corn.so -fPIC -shared $(LNKSTL)

$(BINDIR)/statc.so:	$(STATC_OBJECTS)
	gcc $(STATC_OBJECTS) -o $(BINDIR)/statc.so -fPIC -shared $(LNKSTL)


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
		--exclude OrangeCanvas --exclude GUIApplications --exclude OrangeWidgets ; \
	cd source


clean:
	rm -f $(ODIR)/* $(BINDIR)/*.so $(PPPDIR)/* $(PXDIR)/*
