# Set ORANGE_LIBDIR to some other directory if ../.. is on
# a vfat file-system (not common, but I need it)
# (This is needed because vfat doesn't support symbolic links)
ifndef ORANGE_LIBDIR
  ORANGE_LIBDIR=../..
endif

MODULES=include orange orangeom orangene corn statc

ifndef EXCLUDE_ORANGEQT
MODULES +=  orangeqt
endif

ACTIONS=all clean cleantemp

$(ACTIONS):
	(for MODULE in $(MODULES) ; \
	   do if ! $(MAKE) $@ -C $$MODULE OLD=$(ORANGE_LIBDIR); then exit 1; fi; done)

install: # all orange.pth canvas bininstall docinstall
	@echo "Please use Python Distutils to install Orange."

