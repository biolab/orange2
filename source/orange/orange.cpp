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


//   #pragma warning (disable : 4786 4114 4018 4267 4244 4702 4710 4290)

#include "converts.hpp"
#include "errors.hpp"
#include "root.hpp"
#include "../pyxtract/pyxtract_macros.hpp"
#include "c2py.hpp"
#include "stladdon.hpp"
#include "orange.hpp"

/* Technically, exceptions and warnings belong to entire
   module, not just cls_orange.cpp. On the other hand, they
   need to be in orange_core.

   As orange_core does not (yet) have a general .cpp file,
   I've put this here. */

ORANGE_API PyObject *PyExc_OrangeKernel;
ORANGE_API PyObject *PyExc_OrangeKernelWarning;
ORANGE_API PyObject *PyExc_OrangeAttributeWarning;
ORANGE_API PyObject *PyExc_OrangeWarning;
ORANGE_API PyObject *PyExc_OrangeCompatibilityWarning;

bool initorangeExceptions()
{ if (   ((PyExc_OrangeKernel = makeExceptionClass("orange.KernelException", "An error occurred in Orange's C++ kernel")) == NULL)
      || ((PyExc_OrangeWarning = makeExceptionClass("orange.Warning", "Orange warning", PyExc_Warning)) == NULL)
      || ((PyExc_OrangeCompatibilityWarning = makeExceptionClass("orange.CompatibilityWarning", "Orange compabitility warning", PyExc_OrangeWarning)) == NULL)
      || ((PyExc_OrangeKernelWarning = makeExceptionClass("orange.KernelWarning", "Orange kernel warning", PyExc_OrangeWarning)) == NULL)
      || ((PyExc_OrangeAttributeWarning = makeExceptionClass("orange.AttributeWarning", "A non-builtin attribute has been set", PyExc_OrangeWarning)) == NULL))
    return false;

  TOrange::warningFunction = raiseWarning;

  /* I won't DECREF warningModule (I don't want to unload it)
     filterFunction is borrowed anyway */
  PyObject *warningModule = PyImport_ImportModule("warnings");
  if (!warningModule)
    return false;
  PyObject *filterFunction = PyDict_GetItemString(PyModule_GetDict(warningModule), "filterwarnings");
  if (   !filterFunction
      || !setFilterWarnings(filterFunction, "ignore", ".*", PyExc_OrangeAttributeWarning, "orng.*")
      || !setFilterWarnings(filterFunction, "ignore", "'__callback' is not a builtin attribute of", PyExc_OrangeAttributeWarning, ".*")
      || !setFilterWarnings(filterFunction, "always", ".*", PyExc_OrangeKernelWarning, ".*"))
    return false;
  return true;

}

PyObject *orangeVersion = PyString_FromString("2.0b ("__TIME__", "__DATE__")");

PYCONSTANT(version, orangeVersion)
PYCONSTANT(KernelException, PyExc_OrangeKernel)
PYCONSTANT(AttributeWarning, PyExc_OrangeAttributeWarning)
PYCONSTANT(KernelWarning, PyExc_OrangeKernelWarning)
PYCONSTANT(Warning, PyExc_OrangeWarning)
PYCONSTANT(CompatibilityWarning, PyExc_OrangeCompatibilityWarning)

PYCONSTANT_FLOAT(Illegal_Float, ILLEGAL_FLOAT)


void tdidt_cpp_gcUnsafeInitialization();
void random_cpp_gcUnsafeInitialization();
void pythonVariables_unsafeInitializion();

void gcorangeUnsafeStaticInitialization()
{ tdidt_cpp_gcUnsafeInitialization();
  random_cpp_gcUnsafeInitialization();
  pythonVariables_unsafeInitializion();
}


int PyOr_noConversion(PyObject *obj, void *ptr)
{ return 0; }


void raiseWarning(bool exhaustive, const char *s)
{ if (   (!exhaustive || exhaustiveWarnings)
      && (PyErr_Warn(exhaustive ? PyExc_OrangeCompatibilityWarning : PyExc_OrangeKernelWarning, const_cast<char *>(s))))
    throw mlexception(s);
}


extern char excbuf[256]; // defined in errors.cpp

bool raiseWarning(PyObject *warnType, const char *s, ...)
{ 
  va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, s);
  #else
    va_start(vargs);
  #endif

  vsnprintf(excbuf, 512, s, vargs);

  return PyErr_Warn(warnType, const_cast<char *>(excbuf)) >= 0;
}


bool raiseCompatibilityWarning(const char *s, ...)
{ 
  va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, s);
  #else
    va_start(vargs);
  #endif

  vsnprintf(excbuf, 512, s, vargs);

  return PyErr_Warn(PyExc_OrangeCompatibilityWarning, const_cast<char *>(excbuf)) >= 0;
}


vector<TOrangeType **> classLists;

POrange PyOrType_NoConstructor()
{ throw mlexception("no constructor for this type");
  return POrange();
}

ORANGE_API void addClassList(TOrangeType **classes)
{
  classLists.push_back(classes);
}


TOrangeType *FindOrangeType(const type_info &tinfo)
{ 
  for(vector<TOrangeType **>::const_iterator cli(classLists.begin()), cle(classLists.end()); cli != cle; cli++)
    for(TOrangeType **orty = *cli; *orty; orty++)
      if ((*orty)->ot_classinfo == tinfo)
        return *orty;

  return NULL;
}


bool PyOrange_CheckType(PyTypeObject *pytype)
{ 
  TOrangeType *type = (TOrangeType *)pytype;
  for(vector<TOrangeType **>::const_iterator cli(classLists.begin()), cle(classLists.end()); cli != cle; cli++)
    for(TOrangeType **orty = *cli; *orty; orty++)
      if (*orty == type)
        return true;

  return false;
}


// Ascends the hierarchy until it comes to a class that is from orange's hierarchy
TOrangeType *PyOrange_OrangeBaseClass(PyTypeObject *pytype)
{ 
  while (pytype && !PyOrange_CheckType(pytype))
    pytype=pytype->tp_base;
  return (TOrangeType *)pytype;
}


void *pNotConstructed = malloc(1);

PyObject *WrapWrappedOrange(TWrapped *obj)
{ 
  if (obj==pNotConstructed)
    return PYNULL;

  if (!obj)
    RETURN_NONE;

  if (!obj->myWrapper)
    PYERROR(PyExc_SystemError, "wrong wrapping function called ('WrapOrange' instead of 'WrapNewOrange')", PYNULL);

  PyObject *res = (PyObject *)(obj->myWrapper);

  if (res->ob_type == (PyTypeObject *)&PyOrOrange_Type) {
    PyTypeObject *type = (PyTypeObject *)FindOrangeType(typeid(*obj));
    if (!type) {
      PyErr_Format(PyExc_SystemError, "Orange class '%s' not exported to Python", TYPENAME(typeid(*obj)));
      return PYNULL;
    }
    else
      res->ob_type = type;
  }
      
  Py_INCREF(res);
  return res;
}


bool SetAttr_FromDict(PyObject *self, PyObject *dict, bool fromInit)
{
  if (dict) {
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    char **kc = fromInit ? ((TOrangeType *)(self->ob_type))->ot_constructorkeywords : NULL;
    while (PyDict_Next(dict, &pos, &key, &value)) {
      if (kc) {
        char *kw = PyString_AsString(key);
        char **akc;
        for (akc = kc; *akc && strcmp(*akc, kw); akc++);
        if (*akc)
          continue;
      }
      if (PyObject_SetAttr(self, key, value)<0)
        return false;
    }
  }
  return true;
}


PyObject *yieldNoPickleError(PyObject *self, PyObject *)
{
  PyErr_Format(PyExc_TypeError, "instances of type '%s' cannot be pickled", self->ob_type->tp_name);
  return NULL;
}


PyObject *stringFromList(PyObject *self, TNamedConstantsDef const *ncs)
{
  const long &val = ((PyIntObject *)self)->ob_ival;
  TNamedConstantsDef const *ncsi = ncs;
  for(; ncsi->name && (ncsi->value != val); ncsi++);
  return ncsi->name ? PyString_FromString(ncsi->name) : self->ob_type->tp_base->tp_repr(self);
}


PyObject *unpickleConstant(TNamedConstantRecord const *constList, PyObject *args)
{
  char *s;
  PyObject *args2;
  if (!PyArg_ParseTuple(args, "sO:unpickleConstant", &s, &args2))
    return NULL;
    
  TNamedConstantRecord const *cli = constList;
  for(; cli->name && strcmp(cli->name, s); cli++);
  if (!cli->name)
    PYERROR(PyExc_TypeError, "unpickleConstant: Constant type not found", NULL);
    
  return PyObject_CallObject((PyObject *)cli->type, args2);
}


ORANGE_API PyObject *orangeModule;

#include "orange.px"
#include "initialization.px"
