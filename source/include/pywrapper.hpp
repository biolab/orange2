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


#ifndef __PYWRAPPER_HPP
#define __PYWRAPPER_HPP

#include <math.h>
#include <exception>
using namespace std;

#include "Python.h"

class PyWrapper {
public:
  PyObject *pyobject;

  PyWrapper(PyObject *pyo = NULL)
    : pyobject(pyo)
    { Py_XINCREF(pyobject); }

  ~PyWrapper()
    { Py_XDECREF(pyobject); }

  PyWrapper(const double &f)
    : pyobject(PyFloat_FromDouble(f))
    {}

  PyWrapper(const long &f)
    : pyobject(PyInt_FromLong(f))
    {}

  PyWrapper(const unsigned long &f)
    : pyobject(PyInt_FromLong(f))
    {}

  PyWrapper(const unsigned int &f)
    : pyobject(PyInt_FromLong(f))
    {}

  PyWrapper(const int &f)
    : pyobject(PyInt_FromLong(f))
    {}

  operator PyObject *() const
    { Py_XINCREF(pyobject);
      return pyobject; }

  PyWrapper(const PyWrapper &other)
    : pyobject(other.pyobject)
    { Py_XINCREF(pyobject); }

  void operator = (const PyWrapper &other)
    { inplace(other.pyobject);
    }

  void inplace(PyObject *other)
    { Py_XINCREF(other);
      Py_XDECREF(pyobject);
      pyobject=other;
    }

  int compare(const PyWrapper &other) const
    { int cmp=PyObject_Compare(pyobject, other.pyobject);
      checkForError();
      return cmp;
    }

  bool operator < (const PyWrapper &other) const
    { return compare(other)<0; }

  bool operator <= (const PyWrapper &other) const
    { return compare(other)<=0; }

  bool operator > (const PyWrapper &other) const
    { return compare(other)>0; }

  bool operator >= (const PyWrapper &other) const
    { return compare(other)>=0; }

  bool operator == (const PyWrapper &other) const
    { return !compare(other); }


  PyWrapper operator - () const
    { if (!pyobject)
        return PyWrapper();

      return PyWrapper(PyNumber_Negative(pyobject));
    }

  PyWrapper operator + (const PyWrapper &other) const
    { if (!pyobject)
        return PyWrapper(other.pyobject);

      if (!other.pyobject)
        return PyWrapper(pyobject);

      if (PySequence_Check(pyobject)) 
        return PyWrapper(PySequence_Concat(pyobject, other.pyobject));

      return PyWrapper(PyNumber_Add(pyobject, other));
    }

  PyWrapper operator - (const PyWrapper &other) const
    { if (!pyobject)
        return other.operator -();

      if (!other.pyobject)
        return PyWrapper(pyobject);

      return PyWrapper(PyNumber_Subtract(pyobject, other));
    }

  PyWrapper operator * (const PyWrapper &other) const
    { if (!pyobject || !other.pyobject)
        return PyWrapper();

      return PyWrapper(PyNumber_Multiply(pyobject, other));
    }

  PyWrapper operator / (const PyWrapper &other) const
    { if (!pyobject || !other.pyobject)
        return PyWrapper();

      return PyWrapper(PyNumber_Divide(pyobject, other));
    }

  PyWrapper &operator += (const PyWrapper &other)
    { if (!pyobject) {
        inplace(other.pyobject);
        return *this;
      }

      if (!other.pyobject)
        return *this;

      PyObject *res;
      if (PySequence_Check(pyobject)) 
        res=PySequence_Concat(pyobject, other.pyobject);
      else
        res=PyNumber_Add(pyobject, other);

      inplace(res);
      checkForError();
      Py_XDECREF(res);
      return *this;
    }

  PyWrapper &operator -= (const PyWrapper &other)
    { if (!pyobject) {
        PyWrapper otm=other.operator -();
        inplace(other.pyobject);
        return *this;
      }

      if (!other.pyobject)
        return *this;

      PyObject *res=PyNumber_Subtract(pyobject, other);
      inplace(res);
      Py_XDECREF(res);
      return *this;
    }

  PyWrapper &operator *= (const PyWrapper &other)
    { if (!pyobject || !other.pyobject) {
        inplace(NULL);
        return *this;
      }

      PyObject *res=PyNumber_Multiply(pyobject, other);
      inplace(res);
      Py_XDECREF(res);
      return *this;
    }

  PyWrapper &operator /= (const PyWrapper &other)
    { if (!pyobject || !other.pyobject) {
        inplace(NULL);
        return *this;
      }

      PyObject *res=PyNumber_Divide(pyobject, other);
      inplace(res);
      Py_XDECREF(res);
      return *this;
    }

  PyWrapper &operator ++ ()
    { if (!pyobject)
        return *this;
    
      PyObject *res=PyNumber_Add(pyobject, PyInt_FromLong(1));
      inplace(res);
      Py_XDECREF(res);
      return *this;
    }


  static void checkForError()
  {
    if (PyErr_Occurred())
        throw pyexception();
  }
};

#define UNARY_FUNCTION(name) \
  inline PyWrapper name(const PyWrapper &x) \
  { if (!x.pyobject) \
    throw pyexception("NULL object"); \
\
    PyObject *num=PyNumber_Float(x); \
    if (!num) PyWrapper::checkForError(); \
    return PyWrapper(num ? PyFloat_FromDouble(name(PyFloat_AsDouble(num))) : NULL); \
  }

UNARY_FUNCTION(log)
UNARY_FUNCTION(exp)
UNARY_FUNCTION(sqrt)

#undef UNARY_FUNCTION

inline PyWrapper abs(const PyWrapper &x)
{ if (!x.pyobject)
    throw pyexception("NULL object");

  PyObject *res=PyNumber_Absolute(x);
  if (!res) PyWrapper::checkForError();
  return PyWrapper(res);
}

inline PyWrapper fabs(const PyWrapper &x)
{ if (!x.pyobject)
    throw pyexception("NULL object");

  PyObject *res=PyNumber_Absolute(x);
  if (!res) PyWrapper::checkForError();
  return PyWrapper(res);
}


inline int convert_to_int (const PyWrapper &x)
{ if (!x.pyobject)
    throw pyexception("NULL object");

  PyObject *pyn=PyNumber_Int(x.pyobject);
  PyWrapper::checkForError();

  return int(PyInt_AsLong(pyn));
}

inline double convert_to_double (const PyWrapper &x)
{ if (!x.pyobject)
    throw pyexception("NULL object");

  PyObject *pyn=PyNumber_Float(x.pyobject);
  PyWrapper::checkForError();

  return PyFloat_AsDouble(pyn);
}

// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX

#endif
