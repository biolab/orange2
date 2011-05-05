/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "c2py.hpp"

/* This function is copied from Python's exception.c and (heavily) modified. */
PyObject *makeExceptionClass(char *name, char *docstr, PyObject *base)
{
  PyObject *dict = PyDict_New();
  PyObject *str = NULL;
  PyObject *klass = NULL;

  if (   dict
      && (!docstr ||    ((str = PyString_FromString(docstr)) != NULL)
                     && (!PyDict_SetItemString(dict, "__doc__", str))))
     klass = PyErr_NewException(name, base, dict);

  Py_XDECREF(dict);
  Py_XDECREF(str);
  return klass;
}


bool setFilterWarnings(PyObject *filterFunction, char *action, char *message, PyObject *warning, char *moduleName)
{
  PyObject *args = Py_BuildValue("ssOs", action, message, warning, moduleName);
  PyObject *res = PyObject_CallObject(filterFunction, args);
  Py_DECREF(args);
  if (!res)
    return false;

  Py_DECREF(res);
  return true;
}

