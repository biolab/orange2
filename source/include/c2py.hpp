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

#include <vector> 
#include "stladdon.hpp"


#define PYERROR(type,message,result) \
  { PyErr_SetString(type, message); return result; }
  
#define BREAKPOINT _asm { int 3 }

#define RETURN_NONE { Py_INCREF(Py_None); return Py_None; }
#define NOT_EMPTY(x) (x && (PyDict_Size(x)>0))


inline bool PyFloat_Check_function(PyObject *o)
{ return PyFloat_Check(o); }


inline bool PyInt_Check_function(PyObject *o)
{ return PyInt_Check(o); }


inline bool PyLong_Check_function(PyObject *o)
{ return PyLong_Check(o); }


inline bool PyString_Check_function(PyObject *o)
{ return PyString_Check(o); }


inline float PyNumber_AsFloat(PyObject *o)
{ PyObject *number=PyNumber_Float(o);
  float res=(float)PyFloat_AsDouble(number);
  Py_DECREF(number);
  return res;
}


inline bool convertFromPython(PyObject *obj, int &i)
{ if (PyInt_Check(obj))
    i=(int)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(int)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


inline PyObject *convertToPython(const int &i)
{ return PyInt_FromLong(i); }


inline string convertToString(const int &i)
{ char is[128];
  sprintf(is, "%d", i);
  return is; }



inline bool convertFromPython(PyObject *obj, long &i)
{ if (PyInt_Check(obj))
    i=(long)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(long)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


inline PyObject *convertToPython(const long &i)
{ return PyInt_FromLong(i); }


inline string convertToString(const long &i)
{ char is[128];
  sprintf(is, "%d", int(i));
  return is; }


inline bool convertFromPython(PyObject *obj, unsigned char &i)
{ if (PyInt_Check(obj))
    i=(unsigned char)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(unsigned char)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


inline PyObject *convertToPython(const unsigned char &i)
{ return PyInt_FromLong(i); }


inline string convertToString(const unsigned char &i)
{ char is[128];
  sprintf(is, "%d", i);
  return is; }


inline bool convertFromPython(PyObject *obj, float &i)
{ if (PyFloat_Check(obj))
    i=(float)PyFloat_AsDouble(obj);
  else if (PyInt_Check(obj))
    i=(float)PyFloat_AsDouble(obj);
  else PYERROR(PyExc_TypeError, "invalid number", false);

  return true;
}


inline PyObject *convertToPython(const float &i)
{ return PyFloat_FromDouble(i); }


inline string convertToString(const float &i)
{ char is[128];
  sprintf(is, "%f", i);
  return is; }


inline bool convertFromPython(PyObject *obj, pair<float, float> &i)
{ return PyArg_ParseTuple(obj, "ff", &i.first, &i.second) != 0; }


inline PyObject *convertToPython(const pair<float, float> &i)
{ return Py_BuildValue("ff", i.first, i.second); }


inline string convertToString(const pair<float, float> &i)
{ char is[128];
  sprintf(is, "(%5.3f, %5.3f)", i.first, i.second);
  return is; 
}


inline bool convertFromPython(PyObject *obj, string &str)
{ if (!PyString_Check(obj))
    PYERROR(PyExc_TypeError, "invalid string", false);
  str=PyString_AsString(obj);
  return true;
}


inline PyObject *convertToPython(const string &str)
{ return PyString_FromString(str.c_str()); }


inline string convertToString(const string &str)
{ return str; }


template<class T, class F>
bool PyList2Vector(PyObject *args, vector<T> &floats, bool check(PyObject *), F convert(PyObject *))
{ if (!PyList_Check(args))
    PYERROR(PyExc_TypeError, "invalid type: lst expected", false);

  int size=PyList_Size(args);
  for(int i=0; i<size; i++) {
    PyObject *obj=PyList_GetItem(args, i);
    if (check(obj))
      floats.push_back(T(convert(obj)));
    else {
      floats.clear();
      PYERROR(PyExc_TypeError, "invalid type in lst (floats expected)", false);
    }
  }

  return true;
}


template<class T, class F>
PyObject *Vector2PyList(const vector<T> &floats, PyObject *convert(F))
{ typedef typename vector<T>::const_iterator const_iterator;
  PyObject *lst=PyList_New(floats.size());
  int i=0;
  for(const_iterator fi(floats.begin()), fe(floats.end()); fi!=fe; fi++)
    PyList_SetItem(lst, i++, convert(F(*fi)));
  return lst;
}


inline bool PyList2FloatVector(PyObject *args, vector<float> &floats)
{ return PyList2Vector(args, floats, PyFloat_Check_function, PyFloat_AsDouble); }


inline PyObject *FloatVector2PyList(const vector<float> &floats)
{ return Vector2PyList(floats, PyFloat_FromDouble); }


inline bool PyList2IntVector(PyObject *args, vector<int> &floats)
{ return PyList2Vector(args, floats, PyInt_Check_function, PyInt_AsLong); }


inline PyObject *IntVector2PyList(const vector<int> &floats)
{ return Vector2PyList(floats, PyInt_FromLong); }


inline bool PyList2LongVector(PyObject *args, vector<long> &floats)
{ return PyList2Vector(args, floats, PyInt_Check_function, PyInt_AsLong); }


inline PyObject *LongVector2PyList(const vector<long> &floats)
{ return Vector2PyList(floats, PyInt_FromLong); }


inline bool PyList2StringVector2(PyObject *args, vector<string> &s)
{ return PyList2Vector(args, s, PyString_Check_function, PyString_AsString); }


inline PyObject *richcmp_from_sign(const int &i, const int &op)
{ int cmp;
  switch (op) {
		case Py_LT: cmp = (i<0); break;
		case Py_LE: cmp = (i<=0); break;
		case Py_EQ: cmp = (i==0); break;
		case Py_NE: cmp = (i!=0); break;
		case Py_GT: cmp = (i>0); break;
		case Py_GE: cmp = (i>=0); break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  
  PyObject *res;
  if (cmp)
    res = Py_True;
  else
    res = Py_False;
  Py_INCREF(res);
  return res;
}


template<class T>
PyObject *convertToPython(const T &);


template<class T>
string convertToString(const T &);


class pyexception : public exception {
public:
   PyObject *type, *value, *tracebk;

   pyexception(PyObject *atype, PyObject *avalue, PyObject *atrace)
    : type(atype), value(avalue), tracebk(atrace)
    {}

   pyexception()
    { PyErr_Fetch(&type, &value, &tracebk); }

   pyexception(const string &des)
    : type(PyExc_Exception), value(PyString_FromString(des.c_str())), tracebk(NULL)
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
