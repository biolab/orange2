# Set ORANGE_LIBDIR to some other directory if ../.. is on
# a vfat file-system (not common, but I need it)
# (This is needed because vfat doesn't support symbolic links)
ifndef ORANGE_LIBDIR
  ORANGE_LIBDIR=../..
endif

################################
# External library configuration
################################
# 
#  The folowing options if un-commented (or passes to the make command)
# will cause the build process to link to the corresponding external
# library. The option must name the library to link with. The library 
# and its header files must be on the search path (add the apropriate
# paths to CFLAGS, CXXFLAGS and LDFLAGS if they are not on the default
# search path).
#

######
# BLAS
######

#BLAS_LIB = blas


#######
#LibSVM
#######

#LIBSVM_LIB = svm


###########
# LIBLINEAR
###########

#LIBLINEAR_LIB = linear


#######
# QHULL
#######

#QHULL_LIB = qhull
 

PYTHON ?= $(shell which python)

MODULES=include orange orangeom orangene corn statc

# This is nedded for orangeqt on Mac
ifndef DESTDIR
DESTDIR = $(shell $(PYTHON) -s -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")/site-packages/orange
endif # DESTDIR

ACTIONS=all clean cleantemp

$(ACTIONS):
	(for MODULE in $(MODULES) ; \
	   do if ! $(MAKE) $@ -C $$MODULE OLD=$(ORANGE_LIBDIR) PYTHON=$(PYTHON); then exit 1; fi; done)
ifndef EXCLUDE_ORANGEQT
	if ! $(MAKE) $@ -C orangeqt OLD=$(ORANGE_LIBDIR) PYTHON=$(PYTHON) DESTDIR=$(DESTDIR); then exit 1; fi;
endif	

install: # all orange.pth canvas bininstall docinstall
	@echo "Please use Python Distutils to install Orange."

