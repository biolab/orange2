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


#include "c2py.hpp"

#include "converts.hpp"

bool convertFromPython(PyObject *obj, bool &b)
{ b = PyObject_IsTrue(obj) ? true : false;
  return true;
}


PyObject *convertToPython(const bool &b)
{ return PyInt_FromLong(b ? 1 : 0); }



bool convertFromPython(PyObject *obj, int &i)
{ if (PyInt_Check(obj))
    i=(int)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(int)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


PyObject *convertToPython(const int &i)
{ return PyInt_FromLong(i); }


string convertToString(const int &i)
{ char is[128];
  sprintf(is, "%d", i);
  return is; }



bool convertFromPython(PyObject *obj, long &i)
{ if (PyInt_Check(obj))
    i=(long)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(long)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


PyObject *convertToPython(const long &i)
{ return PyInt_FromLong(i); }


string convertToString(const long &i)
{ char is[128];
  sprintf(is, "%d", int(i));
  return is; }


bool convertFromPython(PyObject *obj, unsigned char &i)
{ if (PyInt_Check(obj))
    i=(unsigned char)PyInt_AsLong(obj);
  else if (PyLong_Check(obj))
    i=(unsigned char)PyLong_AsLong(obj);
  else
    PYERROR(PyExc_TypeError, "invalid integer", false);

  return true;
}


PyObject *convertToPython(const unsigned char &i)
{ return PyInt_FromLong(i); }


string convertToString(const unsigned char &i)
{ char is[128];
  sprintf(is, "%d", i);
  return is; }


bool convertFromPython(PyObject *obj, float &i)
{ if (PyFloat_Check(obj))
    i=(float)PyFloat_AsDouble(obj);
  else if (PyInt_Check(obj))
    i=(float)PyFloat_AsDouble(obj);
  else PYERROR(PyExc_TypeError, "invalid number", false);

  return true;
}


PyObject *convertToPython(const float &i)
{ return PyFloat_FromDouble(i); }


string convertToString(const float &i)
{ char is[128];
  sprintf(is, "%f", i);
  return is; }


bool convertFromPython(PyObject *obj, pair<float, float> &i)
{ return PyArg_ParseTuple(obj, "ff", &i.first, &i.second) != 0; }


PyObject *convertToPython(const pair<float, float> &i)
{ return Py_BuildValue("ff", i.first, i.second); }


string convertToString(const pair<float, float> &i)
{ char is[128];
  sprintf(is, "(%5.3f, %5.3f)", i.first, i.second);
  return is; 
}


bool convertFromPython(PyObject *obj, pair<int, float> &i)
{ return PyArg_ParseTuple(obj, "if", &i.first, &i.second) != 0; }


PyObject *convertToPython(const pair<int, float> &i)
{ return Py_BuildValue("if", i.first, i.second); }


string convertToString(const pair<int, float> &i)
{ char is[128];
  sprintf(is, "(%i, %5.3f)", i.first, i.second);
  return is; 
}


bool convertFromPython(PyObject *obj, string &str)
{ if (!PyString_Check(obj))
    PYERROR(PyExc_TypeError, "invalid string", false);
  str=PyString_AsString(obj);
  return true;
}


PyObject *convertToPython(const string &str)
{ return PyString_FromString(str.c_str()); }


string convertToString(const string &str)
{ return str; }


bool PyNumber_ToFloat(PyObject *o, float &res)
{ PyObject *number=PyNumber_Float(o);
  if (!number) {
    PyErr_Clear();
    return false;
  }
  res = (float)PyFloat_AsDouble(number);
  Py_DECREF(number);
  return true;
}


bool PyNumber_ToDouble(PyObject *o, double &res)
{ PyObject *number=PyNumber_Float(o);
  if (!number) {
    PyErr_Clear();
    return false;
  }
  res = PyFloat_AsDouble(number);
  Py_DECREF(number);
  return true;
}


PyObject *convertToPython(const vector<int> &v)
{
  const int e = v.size();
  PyObject *res = PyList_New(e);
  vector<int>::const_iterator vi(v.begin());
  for(int i = 0; i<e; i++, vi++)
    PyList_SetItem(res, i, PyInt_FromLong(*vi));
  return res;
}


int getBool(PyObject *arg, void *isTrue)
{ 
  int it = PyObject_IsTrue(arg);
  if (it == -1)
    return 0;

  *(bool *)isTrue = it != 0;
  return 1;
}
