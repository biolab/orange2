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


#ifndef __CLS_ORANGE_HPP
#define __CLS_ORANGE_HPP

#include <list>
#include <vector>
#include <typeinfo>

#include "garbage.hpp"
#include "root.hpp"
#include "orange.hpp"
#include "errors.hpp"

PyObject *PyOrType_GenericNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject *PyOrType_GenericNamedNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
PyObject *PyOrType_GenericCallableNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
POrange PyOrType_NoConstructor();

#define SETATTRIBUTES if (!SetAttr_FromDict(self, keywords)) return PYNULL;


PyObject *WrapOrange(POrange);

// This is to be used on just constructed objects only
template<class T>
PyObject *WrapNewOrange(T *obj, PyTypeObject *type)
{ if (!obj) {
    PyErr_Format(PyExc_SystemError, "Constructor for '%s' failed", typeid(*obj).name()+7);
    return PYNULL;
  }
  return WrapOrange(POrange(obj, type));
}

inline TOrangeType *FindOrangeType(POrange obj)
{ return FindOrangeType(typeid(*obj.counter->ptr)); }


//XXX Here, you must force GCPtr to use the correct constructor
#define PyOrange_AS_Orange(op) (GCPtr<TOrange>((TPyOrange *)op, true))

template<class T>
inline T &cast_to(PyObject *op, T *)
{ T *ret = dynamic_cast<T *>(((TPyOrange *)op)->ptr);
  if (!ret)
    raiseError("invalid (null or wrong type) pointer to '%s'", typeid(T).name());
  return *ret;
}


#define PyOrange_AS(type, op) (cast_to(op, (type *)NULL))
#define SELF_AS(type)         (cast_to(self, (type *)NULL))

#define DEFINE_cc(type) \
int cc_##type(PyObject *obj, void *ptr) \
{ if (!PyOr##type##_Check(obj)) \
    return 0; \
  *(GCPtr< T##type > *)(ptr) = PyOrange_As##type(obj); \
  return 1; \
} \
\
int ccn_##type(PyObject *obj, void *ptr) \
{ if (obj == Py_None) {\
    *(GCPtr< T##type > *)(ptr) = GCPtr< T##type >(); \
    return 1; \
  } \
  else \
    return cc_##type(obj, ptr); \
}

#define LIST_PLUGIN_METHODS(name) name##_plugin_methods

#define NAME_CAST_TO_err(type, aname, obj, errreturn) \
  type *obj; \
  PyOrange_AS_Orange(aname).dynamic_cast_to(obj); \
  if (!obj) {\
    if (aname && ((TPyOrange *)aname)->ptr) \
      PyErr_Format(PyExc_TypeError, "invalid object type (expected '%s', got '%s')", typeid(type).name()+7, typeid(*((TPyOrange *)(aname))->ptr).name()+7); \
    else \
      PyErr_Format(PyExc_TypeError, "invalid object type (expected '%s', got nothing)", typeid(type).name()+7); \
    return errreturn; \
  }

#define NAME_CAST_TO(type, name, obj)     NAME_CAST_TO_err(type, name, obj, PYNULL)
#define CAST_TO_err(type, obj, errreturn) NAME_CAST_TO_err(type, self, obj, errreturn)
#define CAST_TO(type, obj)                NAME_CAST_TO_err(type, self, obj, PYNULL)

#endif
