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

