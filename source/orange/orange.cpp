/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifdef _MSC_VER
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
  #include <windows.h>
#endif

#include "orange.hpp"
#include "errors.hpp"
#include "values.hpp"
#include "initialization.px"

TOrangeType *FindOrangeType(const type_info &tinfo)
{ TOrangeType **orty=orangeClasses;
  while (*orty && ((*orty)->ot_classinfo!=tinfo))
    orty++;

  return *orty;
}

bool PyOrange_CheckType(PyTypeObject *pytype)
{ TOrangeType *type=(TOrangeType *)pytype;
  for(TOrangeType **orty=orangeClasses; *orty; orty++)
    if (*orty==type)
      return true;
  return false;
}


// Ascends the hierarchy until it comes to a class that is from orange's hierarchy
TOrangeType *PyOrange_OrangeBaseClass(PyTypeObject *pytype)
{ while (pytype && !PyOrange_CheckType(pytype))
    pytype=pytype->tp_base;
  return (TOrangeType *)pytype;
}


bool SetAttr_FromDict(PyObject *self, PyObject *dict, bool fromInit)
{
  if (dict) {
    int pos = 0;
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


#include "converts.hpp"

void tdidt_cpp_gcUnsafeInitialization();
void random_cpp_gcUnsafeInitialization();

void gcUnsafeStaticInitialization()
{ tdidt_cpp_gcUnsafeInitialization();
  random_cpp_gcUnsafeInitialization();
}


PyObject *PyExc_OrangeKernel;
PyObject *PyExc_OrangeWarning;
PyObject *PyExc_OrangeKernelWarning;
PyObject *PyExc_OrangeAttributeWarning;

void raiseWarning(const char *s)
{ if (PyErr_Warn(PyExc_OrangeKernelWarning, const_cast<char *>(s)))
    throw mlexception(s);
}


bool initExceptions()
{ if (   ((PyExc_OrangeKernel = makeExceptionClass("orange.KernelException", "An error occurred in Orange's C++ kernel")) == NULL)
      || ((PyExc_OrangeWarning = makeExceptionClass("orange.Warning", "Orange warning", PyExc_Warning)) == NULL)
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


PyObject *orangeModule;

void initorange()
{ 
  if (!initExceptions())
    return;

  for(TOrangeType **type=orangeClasses; *type; type++)
    if (PyType_Ready((PyTypeObject *)*type)<0)
      return;

  gcUnsafeStaticInitialization();

  orangeModule = Py_InitModule("orange", orangeFunctions);  
  addConstants(orangeModule);

  PyModule_AddObject(orangeModule, "version", PyString_FromString("0.99b (" __TIME__ ", " __DATE__ ")"));
  PyModule_AddObject(orangeModule, "KernelException", PyExc_OrangeKernel);
  PyModule_AddObject(orangeModule, "AttributeWarning", PyExc_OrangeAttributeWarning);
  PyModule_AddObject(orangeModule, "KernelWarning", PyExc_OrangeKernelWarning);
  PyModule_AddObject(orangeModule, "Warning", PyExc_OrangeWarning);
}


#ifdef _MSC_VER
BOOL APIENTRY DllMain( HANDLE, DWORD  ul_reason_for_call, LPVOID)
{ switch (ul_reason_for_call)
	{ case DLL_PROCESS_ATTACH:case DLL_THREAD_ATTACH:case DLL_THREAD_DETACH:case DLL_PROCESS_DETACH:break; }
  return TRUE;
}
#endif
