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


#ifndef __C2PY_HPP
#define __C2PY_HPP

#include "Python.h"

#include <exception> 
using namespace std;


#define PYERROR(type,message,result) \
  { PyErr_SetString(type, message); return result; }
  
#define BREAKPOINT _asm { int 3 }

#define RETURN_NONE { Py_INCREF(Py_None); return Py_None; }
#define NOT_EMPTY(x) (x && (PyDict_Size(x)>0))


class pyexception : public exception {
public:
   PyObject *type, *value, *tracebk;

   pyexception(PyObject *atype, PyObject *avalue, PyObject *atrace)
    : type(atype), value(avalue), tracebk(atrace)
    {}

   pyexception()
    { PyErr_Fetch(&type, &value, &tracebk); }

   pyexception(const char *des)
    : type(PyExc_Exception), value(PyString_FromString(des)), tracebk(NULL)
    {}
       
   // No destructor! Whoever catches this is responsible to free references
   // (say by calling restore() that passes them on to PyErr_Restore
   
   virtual const char* what () const throw ()
    { PyObject *str=PyObject_Str(value);
      if (str) return PyString_AsString(str); 
      else return "Unidentified Python exception"; }

   void restore()
    { PyErr_Restore(type, value, tracebk); }
};


// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX

PyObject *makeExceptionClass(char *name, char *docstr = NULL, PyObject *base = NULL);
bool setFilterWarnings(PyObject *filterFunction, char *action, char *moduleName, PyObject *warning);

#endif
