Building
========

The documentation can be built with Sphinx 1.0 or newer. Download it at
http://sphinx.pocoo.org/. Also, Orange needs to be installed to build the
documentation. And numpydoc Sphinx extension. To build the documentation, run

    make html

which will create a directory "html" containing the documentation. If make is
not installed on your machine, run

    sphinx-build -b html rst html

in docs/reference. The last two parameters are the input and output directory.

Structure
=========

The actual documentation is intermixed from docs/reference/rst and documented
Python modules in orange/Orange. 

Example scripts and datasets are in docs/reference/rst/code. Additional files,
such as images, are in docs/reference/rst/files.
