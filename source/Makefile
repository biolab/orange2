# Set ORANGE_LIBDIR to some other directory if ../.. is on
# a vfat file-system (not common, but I need it)
# (This is needed because vfat doesn't support symbolic links)
ifndef ORANGE_LIBDIR
  ORANGE_LIBDIR=../..
endif

MODULES=include orange orangeom orangene corn statc
ACTIONS=all clean cleantemp
	
$(ACTIONS):
	(for MODULE in $(MODULES) ; \
	   do make $@ -C $$MODULE OLD=$(ORANGE_LIBDIR); done)

	   
orange.pth:
	@echo "orange" > orange.pth
	@echo "orange\\OrangeWidgets" >> orange.pth
	@echo "orange\\OrangeCanvas" >> orange.pth

canvas:
	@echo "exec `which python` $(DESTDIR)/OrangeCanvas/orngCanvas.pyw" > canvas

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
		for f in `find . -type f -name '*.py' -name '*.pyw' -name '*.png' -o -name '*.pyc'`; \
		do \
			$(INSTALL_DATA) $$f $(DESTDIR)/$$i/$$f; \
		done; \
	done
	$(INSTALL_DATA) ../*.py $(DESTDIR)
	$(INSTALL_DATA) ../*.pyw $(DESTDIR)
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

