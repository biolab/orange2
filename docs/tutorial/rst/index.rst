###############
Orange Tutorial
###############

This is a gentle introduction on scripting in Orange. Orange is a  `Python <http://www.python.org/>`_ library, and the tutorial is a guide through Orange scripting in this language.

We here assume you have already `downloaded and installed Orange <http://orange.biolab.si/download/>`_ and have a working version of Python. Python scripts can run in a terminal window, integrated environments like `PyCharm <http://www.jetbrains.com/pycharm/>`_ and `PythonWin <http://wiki.python.org/moin/PythonWin>`_,
or shells like `iPython <http://ipython.scipy.org/moin/>`_. Whichever environment you are using, try now to import Orange. Below, we used a Python shell::

   % python
   >>> import Orange
   >>> Orange.version.version
   '2.6a2.dev-a55510d'
   >>>

If this leaves no error and warning, Orange and Python are properly
installed and you are ready to continue with this Tutorial.

********
Contents
********

.. toctree::
   :maxdepth: 1

   data.rst
   classification.rst
   regression.rst
   ensembles.rst
   python-learners.rst

****************
Index and Search
****************

* :ref:`genindex`
* :ref:`search`
