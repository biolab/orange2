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


#ifndef __CONVERTS_HPP
#define __CONVERTS_HPP

// here, T must be derived from TOrange
#define _DEFINE_CONVERTFROMPYTHON(_T) \
{ if (allowNull && (!obj || (obj==Py_None))) { \
    var=GCPtr<T##_T>(); \
    return true; \
  } \
  if (!type) \
    type = (PyTypeObject *)FindOrangeType(typeid(T##_T)); \
 \
  if (!obj || !PyObject_TypeCheck(obj, type)) { \
    PyErr_Format(PyExc_TypeError, "expected '%s', got '%s'", type->tp_name, obj ? obj->ob_type->tp_name : "None"); \
    return false; \
  } \
\
   var=GCPtr<T##_T>(PyOrange_AS_Orange(obj)); \
  return true; \
}

#define DEFINE_CONVERTFROMPYTHON(_T) \
bool convertFromPython(PyObject *obj, P##_T &var, bool allowNull=false, PyTypeObject *type=NULL) \
_DEFINE_CONVERTFROMPYTHON(_T)

#define DEFINE_CONVERTFROMPYTHON_TYPE(_T) \
bool convertFromPython(PyObject *obj, P##_T &var, bool allowNull=false, PyTypeObject *type=(PyTypeObject *)&PyOr##_T##_Type) \
_DEFINE_CONVERTFROMPYTHON(_T)

#define DEFINE_CONVERTFROMPYTHON_NODEFAULTS(_T) \
bool convertFromPython(PyObject *obj, P##_T &var, bool allowNull, PyTypeObject *type) \
_DEFINE_CONVERTFROMPYTHON(_T)


#include "externs.px"
#include "garbage.hpp"
#include "Python.h"
WRAPPER(Contingency)
WRAPPER(Distribution)
WRAPPER(CostMatrix)

bool convertFromPython(PyObject *, PContingency &,     bool allowNull=false, PyTypeObject *type=(PyTypeObject *)&PyOrContingency_Type);
bool convertFromPython(PyObject *, PCostMatrix &,      bool allowNull=false, PyTypeObject *type=(PyTypeObject *)&PyOrCostMatrix_Type);

bool convertFromPython(PyObject *, string &);
bool convertFromPython(PyObject *, float &);
bool convertFromPython(PyObject *, pair<float, float> &);
bool convertFromPython(PyObject *, int &);
bool convertFromPython(PyObject *, unsigned char &);

PyObject *convertToPython(const string &);
PyObject *convertToPython(const PCostMatrix &);
PyObject *convertToPython(const float &);
PyObject *convertToPython(const pair<float, float> &);
PyObject *convertToPython(const int &);
PyObject *convertToPython(const unsigned char &);

string convertToString(const PDistribution &);
string convertToString(const string &);
string convertToString(const float &);
string convertToString(const pair<float, float> &);
string convertToString(const int &);
string convertToString(const unsigned char &);
string convertToString(const PContingency &);

bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base);

// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX

#endif
