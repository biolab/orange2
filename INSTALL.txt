Building From Source
====================

Prerequisites
-------------

1. C++ tool-chain. Supported compilers are gcc (g++) for Linux and Mac
OS X and Microsoft Visual C++ on Windows (MinGW is not supported at this
time). Clang also works for MAC OS.  [[BR]]

* On Windows install the free [http://www.microsoft.com/express Visual
  Studio Express] 
* On Linux use the distribution package management tools 
* On Mac OSX install the [http://developer.apple.com/xcode XCode
  developer tools]

You will also need python and numpy development header files. They
are included in python and numpy installation on Windows and Mac OSX,
but are in separate packages on Linux, for instance 'python-dev' and
python-numpy-dev' on Debian derived distributions.

=== Prerequisites on Ubuntu ===

The following commands were tested on Ubuntu 12.04 and Ubuntu 14.04
(requires administrative privileges).

    apt-get update
    apt-get install python-numpy libqt4-opengl-dev libqt4-dev cmake qt4-qmake python-sip-dev python-qt4 python-qt4-dev python-qwt5-qt4 python-sip graphviz python-networkx python-imaging python-qt4-gl build-essential python-pip python-scipy python-pyparsing ipython python-matplotlib
    easy_install -U distribute # optional on Ubuntu 14.04

=== Prerequisites on Fedora ===

The following commands were tested on Fedore 20 (requires administrative
privileges).

    yum update
    yum groupinstall "KDE Software Development"
    yum install numpy scipy python-matplotlib ipython python-pandas sympy python-nose pyparsing python-pip gcc gcc-c++ python-networkx PyQwt

Furthermore, before compilation you will have to modify your PATH.

    export PATH=$PATH:/usr/lib64/qt4/bin/:/usr/lib/qt4/bin/


Obtaining source
----------------

Either:

* Download the latest nightly packed sources archive and unpack it.

* Clone the Mercurial repository:

    hg clone https://bitbucket.org/biolab/orange


Build and Install
-----------------

=== With pip (suggested) ===

Installing with pip automatically downloads the package and builds it provided
that you installed the relevant dependencies.

    sudo pip install --global-option="build_pyqt_ext" orange

=== With setup.py (for developers) ===

The easiest way to build Orange from source is to use the setup.py in
the root Orange directory (requires administrative privileges). Just run:

    python setup.py build_pyqt_ext
    python setup.py build
    python setup.py install 

It the first command fails the scripting interface will still work
perfectly, but some widgets will be missing.

Alternatively, you could install Orange
into the user specific site-packages (see
http://docs.python.org/install/index.html#how-installation-works).
You will need to add the install dir to PYTHONSITEPACKAGES environment
variable (because Python needs to process Orange's .pth file). You can
customize the build process by editing the setup-site.cfg file in this
directory (see the comments in that file for instructions on how to
do that).

After Orange is installed, you can check if everything is working OK by
running the included tests:

    python setup.py test

This command runs all the unit tests and documentation examples. Some
of the latter have additional dependencies you can satisfy by installing
matplotlib, PIL and scipy.

To install in development mode (http://packages.python.org/distribute/setuptools.html#development-mode)
run the following command instead of "python setup.py install":

    python setup.py develop

=== Using make (only for C++ developers) ===

This is only useful to developers of the C++ part (this can only build
the extensions in-place and does not support an install command).

First change the working directory to the 'source' sub-directory then
run make:

    cd source
    make

This will build the orange core extensions in the root directory (i.e. the
one you started in).  Useful environment variables:

 * PYTHON - names the python executable for which Orange is being build.
 * CXXFLAGS - flags to pass to C++ compiler
 * CFLAGS - flags to pass to C compiler
 * LDFLAGS - flags to pass to the linker
 * EXCLUDE_ORANGEQT - if set to any value will cause orangeqt to not be build 

See source/orangeqt/README.txt for building orangeqt alone.

== Linking to external libraries  ==

The Orange source includes some third party libraries that are statically
linked into Orange by default:

 * BLAS (a subset)
 * LIBSVM (v3.2 - v3.* is required)
 * LIBLINEAR (v1.8)
 * QHull

To customize the build process to link to the corresponding external
library instead, try.

* For the setup.py method modify the setup-site.cfg file.

* For make, pass the library names on the command line or through
  environment variables (listed in source/Makefile). Example:

    make BLAS_LIB=atlas

