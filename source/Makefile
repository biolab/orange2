PYTHONINCLUDE = $(shell python -c "import sys; print sys.prefix+'/include/python'+sys.version[:3]")
PYTHONLIB = $(shell python -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")
PYTHONSITEPKGS = $(PYTHONLIB)/site-packages

DESTDIR = $(ROOT)$(PYTHONSITEPKGS)/orange
DOCDIR = $(ROOT)/usr/local/doc/orange
SCRIPTDIR = $(ROOT)/usr/local/bin
INSTALL = install
INSTALL_DATA = $(INSTALL) -m 644
INSTALL_SCRIPT = $(INSTALL)
INSTALL_SHARED = $(INSTALL) -m 555
INSTALL_DIR = $(INSTALL) -d -m 755

COMPILEOPTIONS=-fPIC -w -DLINUX -O3 -DNO_NUMERIC
LINKOPTIONS=-shared -lstdc++
export CPATH=orange:orange/ppp:orange/px:include:corn:statc:$(PYTHONINCLUDE):/usr/local/include:/usr/include

vpath %.cpp orange:corn:statc:include
vpath %.hpp orange:corn:statc:include
vpath %.h orange:include:$(PYTHONINCLUDE):/usr/local/include/gsl:/usr/include/gsl
vpath %.px orange/px

obj/%.o :
	g++ $(COMPILEOPTIONS) -c $< -o $@


all	: dirs ../orange.so ../corn.so ../statc.so orange.pth
            
include makefile.deps

dirs:
	-@mkdir -p obj > /dev/null

../orange.so: orange/ppp/stamp orange/px/stamp $(ORANGE_OBJECTS)
	gcc $(ORANGE_OBJECTS) $(LINKOPTIONS) -o ../orange.so -lgsl -lgslcblas

../corn.so:	$(CORN_OBJECTS)
	gcc $(CORN_OBJECTS) $(LINKOPTIONS) -o ../corn.so

../statc.so:	$(STATC_OBJECTS)
	gcc $(STATC_OBJECTS) $(LINKOPTIONS) -o ../statc.so

orange.pth:
	@echo "orange" > orange.pth
	@echo "orange\\OrangeWidgets" >> orange.pth
	@echo "orange\\OrangeCanvas" >> orange.pth

canvas:
	@echo "exec `which python` $(DESTDIR)/OrangeCanvas/orngCanvas.py" > canvas

install: all orange.pth canvas bininstall docinstall

bininstall:
	python -c "import compileall; compileall.compile_dir('..')"
	$(INSTALL_DIR) $(DESTDIR)
	$(INSTALL_DIR) $(SCRIPTDIR)
	@for i in OrangeCanvas OrangeWidgets GUIApplications; \
	do \
		if test ! -d $(DESTDIR)/$$i; then \
			echo "Creating directory $(DESTDIR)/$$i"; \
			$(INSTALL_DIR) $(DESTDIR)/$$i; \
		else	true; \
		fi; \
		cd ../$$i; \
		for s in `find . -type d`; \
		do \
			if test ! -d $(DESTDIR)/$$i/$$s; then \
				echo "Creating subdirectory $(DESTDIR)/$$i/$$s"; \
				$(INSTALL_DIR) $(DESTDIR)/$$i/$$s; \
			else	true; \
			fi; \
		done; \
		for f in `find . -type f -name '*.py' -o -name '*.pyc'`; \
		do \
			$(INSTALL_DATA) $$f $(DESTDIR)/$$i/$$f; \
		done; \
	done
	$(INSTALL_DATA) ../*.py $(DESTDIR)
	$(INSTALL_DATA) ../*.pyc $(DESTDIR)
	$(INSTALL_DATA) ../COPYING $(DESTDIR)
	$(INSTALL_SHARED) ../*.so $(DESTDIR)
	$(INSTALL_DATA) orange.pth $(DESTDIR)/..
	$(INSTALL_SCRIPT) canvas $(SCRIPTDIR)
	cd $(DESTDIR)/OrangeCanvas; \
	python xmlParse.py

docinstall:
	@cd ../doc; \
	$(INSTALL_DIR) $(DOCDIR); \
	for s in `find . -type d`; \
	do \
		if test ! -d $(DOCDIR)/$$s; then \
			echo "Creating subdirectory $(DOCDIR)/$$s"; \
			$(INSTALL_DIR) $(DOCDIR)/$$s; \
		else	true; \
		fi; \
	done; \
	for f in `find . -type f`; \
	do \
		$(INSTALL_DATA) $$f $(DOCDIR)/$$f; \
	done

clean:
	rm -f obj/* ../*.so orange/ppp/* orange/px/* orange.pth canvas
