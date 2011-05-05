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
 

#ifdef _MSC_VER
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
  #include <windows.h>
#endif

#include "statc.hpp"
#include "pywrapper.hpp"
#include "stat.hpp"
#include "stladdon.hpp"
//#include <algorithm>
#include <string>
using namespace std;

/* *********** EXCEPTION CATCHING ETC. ************/

 
#undef min
#undef max
 
#define PyTRY try {
 
#define PYNULL ((PyObject *)NULL)
#define PyCATCH   PyCATCH_r(PYNULL)
#define PyCATCH_1 PyCATCH_r(-1)
 
#define PyCATCH_r(r) \
  } \
catch (pyexception err)   { err.restore(); return r; } \
catch (exception err) { PYERROR(PyExc_StatcKernel, err.what(), r); }

PyObject *PyExc_StatcKernel;
PyObject *PyExc_StatcWarning;

/* *********** MODULE INITIALIZATION ************/

STATC_API void initstatc()
{ if (   ((PyExc_StatcKernel = makeExceptionClass("statc.KernelException", "an error occurred in statc's C++ code")) == NULL)
      || ((PyExc_StatcWarning = makeExceptionClass("statc.Warning", "statc warning", PyExc_Warning)) == NULL))
    return;

  PyObject *me = Py_InitModule("statc", statc_functions);

  PyObject *pdm = PyModule_New("pointDistribution");
  PyModule_AddObject(pdm, "Minimal", PyInt_FromLong(DISTRIBUTE_MINIMAL));
  PyModule_AddObject(pdm, "Factor", PyInt_FromLong(DISTRIBUTE_FACTOR));
  PyModule_AddObject(pdm, "Fixed", PyInt_FromLong(DISTRIBUTE_FIXED));
  PyModule_AddObject(pdm, "Uniform", PyInt_FromLong(DISTRIBUTE_UNIFORM));
  PyModule_AddObject(pdm, "Maximal", PyInt_FromLong(DISTRIBUTE_MAXIMAL));

  PyModule_AddObject(me, "pointDistribution", pdm);
}


#ifdef _MSC_VER
BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{ switch (ul_reason_for_call)
	{ case DLL_PROCESS_ATTACH:case DLL_THREAD_ATTACH:case DLL_THREAD_DETACH:case DLL_PROCESS_DETACH:break; }
  return TRUE;
}
#endif

/* *********** CONVERSION TO AND FROM PYTHON ************/


bool py2double(PyObject *pyo, double &dd)
{ 
  PyObject *pyn=PyNumber_Float(pyo);
  if (!pyn)
    PYERROR(PyExc_TypeError, "invalid number", false);
  dd=PyFloat_AsDouble(pyn);
  Py_DECREF(pyn);
  return true;
}

bool py2int(PyObject *pyo, int &dd)
{ PyObject *pyn=PyNumber_Int(pyo);
  if (!pyn)
    PYERROR(PyExc_TypeError, "invalid number", false);
  dd=int(PyInt_AsLong(pyn));
  Py_DECREF(pyn);
  return true;
}


bool PyList2flist(PyObject *pylist, vector<double> &flist)
{ int len=PyList_Size(pylist);
  flist=vector<double>(len);
  for(int i=0; i<len; i++) {
    PyObject *item=PyList_GetItem(pylist, i);
    PyObject *number=PyNumber_Float(item);
    if (!number)
      PYERROR(PyExc_AttributeError, "invalid number in list", false);
    flist[i]=PyFloat_AsDouble(number);
    Py_DECREF(number);
  }
  return true;
}


bool PyList2flist2d(PyObject *pylist, vector<vector<double> > &flist)
{ int len=PyList_Size(pylist);
  flist=vector<vector<double> >(len);
  for(int i=0; i<len; i++) {
    PyObject *slist=PyList_GetItem(pylist, i);
    if (!PyList_Check(slist))
      PYERROR(PyExc_TypeError, "list expected", false);
    if (!PyList2flist(slist, flist[i]))
      return false;
  }

  return true;
}


bool args2flist(PyObject *args, vector<double> &flist)
{ PyObject *pylist;
  if (   !PyArg_ParseTuple(args, "O", &pylist)
      || !PyList_Check(pylist))
    PYERROR(PyExc_AttributeError, "list expected", false)

  return PyList2flist(pylist, flist);
}


bool args2flist2d(PyObject *args, vector<vector<double> > &flist)
{ PyObject *pylist;
  if (   !PyArg_ParseTuple(args, "O", &pylist)
      || !PyList_Check(pylist))
    PYERROR(PyExc_AttributeError, "list expected", false)

  return PyList2flist2d(pylist, flist);
}

bool args22lists(PyObject *args, vector<double> &flist1, vector<double> &flist2)
{ PyObject *pylist1, *pylist2;
  if (   !PyArg_ParseTuple(args, "OO", &pylist1, &pylist2)
      || !PyList_Check(pylist1)
      || !PyList_Check(pylist2)
      || (PyList_Size(pylist1)!=PyList_Size(pylist2)))
    PYERROR(PyExc_AttributeError, "two lists of equal sizes expected", false)

  return PyList2flist(pylist1, flist1) && PyList2flist(pylist2, flist2);
}

bool args22listsne(PyObject *args, vector<double> &flist1, vector<double> &flist2)
{ PyObject *pylist1, *pylist2;
  if (   !PyArg_ParseTuple(args, "OO", &pylist1, &pylist2)
      || !PyList_Check(pylist1)
      || !PyList_Check(pylist2))
    PYERROR(PyExc_AttributeError, "two lists expected", false)

  return PyList2flist(pylist1, flist1) && PyList2flist(pylist2, flist2);
}


PyObject *flist2PyList(const vector<double> &flist)
{ PyObject *pylist=PyList_New(flist.size());
  int i=0;
  const_ITERATE(vector<double>, fi, flist)
    PyList_SetItem(pylist, i++, PyFloat_FromDouble(*fi));
  return pylist;
}


PyObject *ilist2PyList(const vector<int> &ilist)
{ PyObject *pylist=PyList_New(ilist.size());
  int i=0;
  const_ITERATE(vector<int>, fi, ilist)
    PyList_SetItem(pylist, i++, PyInt_FromLong(*fi));
  return pylist;
}



bool PyList2wlist(PyObject *pylist, vector<PyWrapper> &wlist)
{ int len=PyList_Size(pylist);
  wlist=vector<PyWrapper>();
  wlist.reserve(len);
  for(int i=0; i<len; i++)
    wlist.push_back(PyWrapper(PyList_GetItem(pylist, i)));
  return true;
}


bool PyList2wlist2d(PyObject *pylist, vector<vector<PyWrapper> > &flist)
{ int len=PyList_Size(pylist);
  flist=vector<vector<PyWrapper> >(len);
  for(int i=0; i<len; i++) {
    PyObject *slist=PyList_GetItem(pylist, i);
    if (!PyList_Check(slist))
      PYERROR(PyExc_TypeError, "list expected", false);
    if (!PyList2wlist(slist, flist[i]))
      return false;
  }

  return true;
}


bool args2wlist2d(PyObject *args, vector<vector<PyWrapper> > &wlist)
{ PyObject *pylist;
  if (   !PyArg_ParseTuple(args, "O", &pylist)
      || !PyList_Check(pylist))
    PYERROR(PyExc_AttributeError, "list expected", false)

  return PyList2wlist2d(pylist, wlist);
}


bool args2wlist(PyObject *args, vector<PyWrapper> &wlist)
{ PyObject *pylist;
  if (   !PyArg_ParseTuple(args, "O", &pylist)
      || !PyList_Check(pylist))
    PYERROR(PyExc_AttributeError, "list expected", false)

  return PyList2wlist(pylist, wlist);
}


bool args22wlists(PyObject *args, vector<PyWrapper> &flist1, vector<PyWrapper> &flist2)
{ PyObject *pylist1, *pylist2;
  if (   !PyArg_ParseTuple(args, "OO", &pylist1, &pylist2)
      || !PyList_Check(pylist1)
      || !PyList_Check(pylist2)
      || (PyList_Size(pylist1)!=PyList_Size(pylist2)))
    PYERROR(PyExc_AttributeError, "two lists of equal sizes expected", false)

  return PyList2wlist(pylist1, flist1) && PyList2wlist(pylist2, flist2);
}


bool args22wlistsne(PyObject *args, vector<PyWrapper> &flist1, vector<PyWrapper> &flist2)
{ PyObject *pylist1, *pylist2;
  if (   !PyArg_ParseTuple(args, "OO", &pylist1, &pylist2)
      || !PyList_Check(pylist1)
      || !PyList_Check(pylist2))
    PYERROR(PyExc_AttributeError, "two lists expected", false)

  return PyList2wlist(pylist1, flist1) && PyList2wlist(pylist2, flist2);
}


PyObject *wlist2PyList(const vector<PyWrapper> &flist)
{ PyObject *pylist=PyList_New(flist.size());
  int i=0;
  const_ITERATE(vector<PyWrapper>, fi, flist)
    PyList_SetItem(pylist, i++, *fi);
  return pylist;
}


/* *********** MACROS FOR GENERAL FUNCTION DEFINITIONS ************/

#define T_FROM_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> flist; \
    if (args2flist(args, flist)) \
      return PyFloat_FromDouble(name(flist)); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wlist; \
    if (args2wlist(args, wlist)) \
      return name(wlist); \
    \
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_LIST_optT(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    vector<double> flist; \
    double init=0.0; \
    if (   PyArg_ParseTuple(args, "O|d", &pylist, &init) \
        && PyList2flist(pylist, flist)) \
      return PyFloat_FromDouble(name(flist, init)); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wlist; \
    PyObject *pyinit=NULL; \
    if (   PyArg_ParseTuple(args, "O|O", &pylist, &pyinit) \
        && PyList2wlist(pylist, wlist)) \
      return (PyObject *)(name(wlist, PyWrapper(pyinit))); \
    \
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_LIST_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> x, y; \
    if (args22lists(args, x, y)) \
      return PyFloat_FromDouble(name(x, y)); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wx, wy; \
    if (args22wlists(args, wx, wy)) \
      return name(wx, wy); \
    \
    PYERROR(PyExc_AttributeError, #name": two lists expected", PYNULL); \
  PyCATCH \
}


#define T_FROM_LIST_LIST_optT(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pyx, *pyy; \
    vector<double> x, y; \
    double init=0.0; \
    if (   PyArg_ParseTuple(args, "OO|d", &pyx, &pyy, &init) \
        && PyList2flist(pyx, x)  \
        && PyList2flist(pyy, y)) \
      return PyFloat_FromDouble(name(x, y, init)); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wx, wy; \
    PyObject *pyinit=NULL; \
    if (   PyArg_ParseTuple(args, "OO|d", &pyx, &pyy, &pyinit) \
        && PyList2wlist(pyx, wx)  \
        && PyList2wlist(pyy, wy)) \
      return (PyObject *)(name(wx, wy, PyWrapper(pyinit))); \
    \
    return PYNULL; \
  PyCATCH \
}



#define LIST_FROM_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> x, res; \
    if (args2flist(args, x)) \
      name(x, res); \
      return flist2PyList(res); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> w, wres; \
    if (args2wlist(args, w)) \
      name(w, wres); \
      return wlist2PyList(wres); \
    \
    PYERROR(PyExc_AttributeError, #name": list expected", PYNULL); \
  PyCATCH \
}



#define LIST_FROM_LIST_optT(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    vector<double> flist, res; \
    double init=0.0; \
    if (   PyArg_ParseTuple(args, "O|d", &pylist, &init) \
        && PyList2flist(pylist, flist)) {\
      name(flist, res); \
      return flist2PyList(res); \
    } \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wlist, wres; \
    PyObject *pyinit=NULL; \
    if (   PyArg_ParseTuple(args, "O|O", &pylist, &pyinit) \
        && PyList2wlist(pylist, wlist)) { \
      name(wlist, wres); \
      return wlist2PyList(wlist); \
    } \
    \
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_LIST_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    double mom; \
    vector<double> flist; \
    if (   PyArg_ParseTuple(args, "Od", &pylist, &mom) \
        && PyList2flist(pylist, flist)) \
          return PyFloat_FromDouble(name(flist, mom)); \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wlist; \
    PyObject *wmom; \
    if (   PyArg_ParseTuple(args, "OO", &pylist, &wmom)  \
        && PyList2wlist(pylist, wlist)) \
          return name(wlist, PyWrapper(wmom)); \
    \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}


#define T_FROM_LIST_plus(name, type, pys) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    type mom; \
    if (   PyArg_ParseTuple(args, pys, &pylist, &mom)) {\
      vector<double> flist; \
      if (PyList2flist(pylist, flist)) \
        return PyFloat_FromDouble(name(flist, mom)); \
      \
      PyErr_Clear(); \
      \
      vector<PyWrapper> wlist; \
      if (PyList2wlist(pylist, wlist)) \
        return name(wlist, mom); \
    } \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}

#define T_FROM_LIST_INT(name) T_FROM_LIST_plus(name, int, "Oi")
#define T_FROM_LIST_DOUBLE(name) T_FROM_LIST_plus(name, double, "Od")



#define LIST_FROM_LIST_plus(name, type, pys) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    type mom; \
    if (   PyArg_ParseTuple(args, pys, &pylist, &mom)) {\
      vector<double> flist; \
      if (PyList2flist(pylist, flist)) { \
        vector<double> fres; \
        name(flist, mom, fres); \
        return flist2PyList(fres); \
      } \
      \
      PyErr_Clear(); \
      \
      vector<PyWrapper> wlist; \
      if (PyList2wlist(pylist, wlist)) { \
        vector<PyWrapper> wres; \
        name(wlist, mom, wres); \
        return wlist2PyList(wres); \
      } \
    } \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}

#define LIST_FROM_LIST_INT(name) LIST_FROM_LIST_plus(name, int, "Oi")
#define LIST_FROM_LIST_DOUBLE(name) LIST_FROM_LIST_plus(name, double, "Od")


#define DOUBLE_DOUBLE_FROM_LIST_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> x, y; \
    double res1, res2; \
    if (args22lists(args, x, y)) {\
      res1=name(x, y, res2); \
      return Py_BuildValue("dd", res1, res2); \
    } \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wx, wy; \
    if (args22wlists(args, wx, wy)) {\
      res1=name(wx, wy, res2); \
      return Py_BuildValue("dd", res1, res2); \
    } \
    \
    PYERROR(PyExc_AttributeError, #name": two lists of equal size expected", PYNULL); \
  PyCATCH \
}


#define T_T_FROM_LIST_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> x, y; \
    if (args22lists(args, x, y)) { \
      double res1, res2; \
      res1=name(x, y, res2); \
      return Py_BuildValue("dd", res1, res2); \
    } \
    \
    PyErr_Clear(); \
    \
    vector<PyWrapper> wx, wy; \
    if (args22wlists(args, wx, wy)) { \
      PyWrapper res1, res2; \
      res1=name(wx, wy, res2); \
      return Py_BuildValue("NN", (PyObject *)res1, (PyObject *)res2); \
    } \
    \
    PYERROR(PyExc_AttributeError, #name": two lists of equal size expected", PYNULL); \
  PyCATCH \
}


#define T_T_FROM_LIST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    vector<double> flist; \
    if (!args2flist(args, flist)) \
      return NULL; \
\
      double res1, res2; \
      res1 = name(flist, res2); \
      return Py_BuildValue("dd", res1, res2); \
  PyCATCH \
}


#define T_T_FROM_LIST_plus(name, type, pys) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    type mom; \
    if (   PyArg_ParseTuple(args, pys, &pylist, &mom)) {\
      vector<double> flist; \
      if (PyList2flist(pylist, flist)) \
        double res1, res2; \
        res1=name(flist, mom, res2); \
        return PyBuildValue("dd", res1, res2); \
      \
      PyErr_Clear(); \
      \
      vector<PyWrapper> wlist; \
      if (PyList2wlist(pylist, wlist)) \
        PyWrapper res1, res2; \
        res1=name(wlist, mom, res2); \
        return PyBuildValue("OO", (PyObject *)res1, (PyObject *)res2); \
    } \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}

#define T_T_FROM_LIST_INT(name) T_T_FROM_LIST_plus(name, int, "Oi")
#define T_T_FROM_LIST_DOUBLE(name) T_T_FROM_LIST_plus(name, double, "Od")



#define DOUBLE_DOUBLE_FROM_LIST_plus(name, type, pys) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    double res1, res2; \
    type mom; \
    if (   PyArg_ParseTuple(args, pys, &pylist, &mom)) {\
      vector<double> flist; \
      if (PyList2flist(pylist, flist)) \
        res1=name(flist, mom, res2); \
      \
      PyErr_Clear(); \
      \
      vector<PyWrapper> wlist; \
      if (PyList2wlist(pylist, wlist)) \
        res1=name(wlist, mom, res2); \
      \
      else PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
      \
      return PyBuildValue("dd", res1, res2); \
    } \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}

#define DOUBLE_DOUBLE_FROM_LIST_INT(name) DOUBLE_DOUBLE_FROM_LIST_plus(name, int, "Oi")
#define DOUBLE_DOUBLE_FROM_LIST_DOUBLE(name) DOUBLE_DOUBLE_FROM_LIST_plus(name, double, "Od")


#define T_T_FROM_LIST_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    PyObject *pylist; \
    double mom; \
    vector<double> flist; \
    if (   PyArg_ParseTuple(args, "Od", &pylist, &mom)\
        && (PyList2flist(pylist, flist))) {\
        double res1, res2; \
        res1=name(flist, mom, res2); \
        return Py_BuildValue("dd", res1, res2); \
      } \
      \
      PyErr_Clear(); \
      \
    PyObject *wmom; \
    vector<PyWrapper> wlist; \
    if (   PyArg_ParseTuple(args, "OO", &pylist, &wmom)\
        && (PyList2wlist(pylist, wlist))) {\
        PyWrapper res1, res2; \
        res1=name(wlist, PyWrapper(mom), res2); \
        return Py_BuildValue("NN", (PyObject *)res1, (PyObject *)res2); \
    } \
    PYERROR(PyExc_AttributeError, #name": invalid arguments", PYNULL); \
  PyCATCH \
}


#define T_FROM_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    double x; \
    if (PyArg_ParseTuple(args, "d", &x)) \
      return PyFloat_FromDouble(name(x)); \
\
    PyErr_Clear(); \
\
    PyObject *pyx; \
    if (PyArg_ParseTuple(args, "O", &pyx)) \
      return (PyObject *)(name(PyWrapper(pyx))); \
\
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_T_T_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    double x, y, z; \
    if (PyArg_ParseTuple(args, "ddd", &x, &y, &z)) \
      return PyFloat_FromDouble(name(x, y, z)); \
\
    PyErr_Clear(); \
\
    PyObject *pyx, *pyy, *pyz; \
    if (PyArg_ParseTuple(args, "OOO", &pyx, &pyy, &pyz)) \
      return (PyObject *)(name(PyWrapper(pyx), PyWrapper(pyy), PyWrapper(pyz))); \
\
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_T_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    double x, y; \
    if (PyArg_ParseTuple(args, "dd", &x, &y)) \
      return PyFloat_FromDouble(name(x, y)); \
\
    PyErr_Clear(); \
\
    PyObject *pyx, *pyy; \
    if (PyArg_ParseTuple(args, "OO", &pyx, &pyy)) \
      return (PyObject *)(name(PyWrapper(pyx), PyWrapper(pyy))); \
\
    return PYNULL; \
  PyCATCH \
}


#define T_FROM_INT_INT_T(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    double x; \
    int i1, i2; \
    if (PyArg_ParseTuple(args, "iid", &i1, &i2, &x)) \
      return PyFloat_FromDouble(name(i1, i2, x)); \
\
    PyErr_Clear(); \
\
    PyObject *pyx; \
    if (PyArg_ParseTuple(args, "iiO", &i1, &i2, &pyx)) \
      return (PyObject *)(name(i1, i2, PyWrapper(pyx))); \
\
    return PYNULL; \
  PyCATCH \
}


/* *********** AUXILIARY FUNCTIONS ************/


class Callback {
public:
  PyObject *callback;

  Callback(PyObject *cb)
    : callback(cb)
  { if (cb)
      if (cb==Py_None)
        callback=PYNULL;
      else {
        if (!PyCallable_Check(cb))
          throw StatException("Callback: non-callable callback function");

        Py_XINCREF(callback);
      }
  }

  ~Callback()
  { Py_XDECREF(callback); }

  Callback(const Callback &other)
   : callback(other.callback)
   { Py_XINCREF(callback); }

  void operator = (const Callback &other)
   { Py_XINCREF(other.callback);
     Py_XDECREF(callback);
     callback=other.callback;
   }
};


class BoolUnaryCallback : public Callback {
public:
  BoolUnaryCallback(PyObject *cb)
    : Callback(cb)
    {}

  bool operator ()(const PyWrapper &x) const
  { if (!x.pyobject)
      throw StatException("BoolUnaryCallable: invalid object");

    PyObject *args=Py_BuildValue("(O)", x.pyobject);
    PyObject *res=PyEval_CallObject(callback, args);
    Py_DECREF(args);

    PyWrapper::checkForError();

    return PyObject_IsTrue(res)!=0;
  }
};


class BoolBinaryCallback : public Callback {
public:
  typedef PyWrapper first_argument_type;
  typedef PyWrapper second_argument_type;
  typedef bool result_type;

  BoolBinaryCallback(PyObject *cb)
    : Callback(cb)
    {}

  bool operator ()(const PyWrapper &x, const PyWrapper &y) const
  { if (!x.pyobject || !y.pyobject)
      throw StatException("BoolBinaryCallable: invalid objects");

    PyObject *args=Py_BuildValue("OO", x.pyobject, y.pyobject);
    PyObject *res=PyEval_CallObject(callback, args);
    Py_DECREF(args);

    PyWrapper::checkForError();

    return PyObject_IsTrue(res)!=0;
  }
};

class IsTrueCallback : public BoolUnaryCallback {
public:
    IsTrueCallback(PyObject *cb=NULL)
      : BoolUnaryCallback(cb)
      {}

    bool operator()(const PyWrapper &x) const
    { if (!x.pyobject)
        throw StatException("IsTrueCallback: invalid object");

      return callback ? BoolUnaryCallback::operator ()(x) : (PyObject_IsTrue(x.pyobject)!=0);
    }
};


class LessThanCallback : public BoolBinaryCallback {
public:
    LessThanCallback(PyObject *o=NULL)
      : BoolBinaryCallback(o)
      {}

    bool operator()(const PyWrapper &x, const PyWrapper &y) const
    { if (!x.pyobject || !y.pyobject)
        throw StatException("CompareCallback: invalid objects");
      return callback ? BoolBinaryCallback::operator ()(x, y) : (x<y);
    }
};



/* *********** CENTRAL TENDENCY ************/


T_FROM_LIST(geometricmean)
T_FROM_LIST(harmonicmean)
T_FROM_LIST(mean)


PyObject *py_median(PyObject *, PyObject *args)
{ PyTRY
    vector<double> flist;
    if (args2flist(args, flist))
      return PyFloat_FromDouble(median(flist));

    PyErr_Clear();

    vector<PyWrapper> wlist;
    if (args2wlist(args, wlist))
      return median(wlist);

    PyErr_Clear();

    PyObject *pylist, *pycomp=NULL;
    if (   PyArg_ParseTuple(args, "O|O", &pylist, &pycomp)
        || !PyList_Check(pylist))
      PYERROR(PyExc_AttributeError, "list expected", PYNULL)
    if (!PyList2wlist(pylist, wlist))
      return PYNULL;

    return median(wlist, LessThanCallback(pycomp));

  PyCATCH
}


PyObject *py_mode(PyObject *, PyObject *args)
{ PyTRY
    vector<double> flist;
    if (args2flist(args, flist)) {
      vector<double> modes;
      int res=mode(flist, modes);
      return Py_BuildValue("iN", res, flist2PyList(modes));
    }

    PyErr_Clear();

    PyObject *pylist, *pycomp=NULL;
    vector<PyWrapper> wlist, modes;
    if (   !PyArg_ParseTuple(args, "O|O", &pylist, &pycomp)
        || !PyList_Check(pylist))
      PYERROR(PyExc_AttributeError, "mode: list and optional compare function expected", PYNULL)
    if (!PyList2wlist(pylist, wlist))
      return PYNULL;

    int res= pycomp ? mode(wlist, modes, LessThanCallback(pycomp))
                    : mode(wlist, modes);
    PYERROR(PyExc_SystemError, "mode: failed", PYNULL);
    return Py_BuildValue("iN", res, wlist2PyList(modes));
 
  PyCATCH
}



/* *********** MOMENTS ************/

T_FROM_LIST_INT(moment)
T_FROM_LIST(variation)
T_FROM_LIST(skewness)
T_FROM_LIST(kurtosis)


/* *********** FREQUENCY STATS************/


PyObject *py_scoreatpercentile(PyObject *, PyObject *args)
{ PyTRY
    PyObject *pylist;
    double perc;
    vector<double> flist;
    if (   PyArg_ParseTuple(args, "Od", &pylist, &perc)
        && PyList2flist(pylist, flist))
      return PyFloat_FromDouble(scoreatpercentile(flist, perc));

    PyErr_Clear();

    vector<PyWrapper> wlist;
    PyObject *pycomp=NULL;
    if (   PyArg_ParseTuple(args, "Od|O", &pylist, &perc, &pycomp)
        && PyList2wlist(pylist, wlist))
      return (PyObject *)(scoreatpercentile(wlist, perc, LessThanCallback(pycomp)));

    PYERROR(PyExc_AttributeError, "scoreatpercentile: list, percentile and optional compare function expected", PYNULL);

  PyCATCH
}


PyObject *py_percentileofscore(PyObject *, PyObject *args)
{ PyTRY
    PyObject *pylist;
    double score;
    vector<double> flist;
    if (   PyArg_ParseTuple(args, "Od", &pylist, &score)
        && PyList2flist(pylist, flist))
      return PyFloat_FromDouble(percentileofscore(flist, score));

    PyErr_Clear();

    vector<PyWrapper> wlist;
    PyObject *pyscore, *pycomp=NULL;
    if (   PyArg_ParseTuple(args, "OO|O", &pylist, &pyscore, &pycomp)
        && PyList2wlist(pylist, wlist))
      return PyFloat_FromDouble(percentileofscore(wlist, PyWrapper(pyscore), LessThanCallback(pycomp)));

    PYERROR(PyExc_AttributeError, "percentileofscore: list, score and optional compare function expected", PYNULL);

  PyCATCH
}


#define CALL_HISTOGRAM 0
#define CALL_CUMFREQ   1
#define CALL_RELFREQ   2

#define HISTOSWITCH(pars, parsd) \
  switch (function) { \
    case CALL_HISTOGRAM: histogram pars; break; \
    case CALL_CUMFREQ: cumfreq pars; break; \
    case CALL_RELFREQ: relfreq parsd; break; \
  }

PyObject *py_histograms(PyObject *args, int function)
{ PyTRY
    vector<int> counts;
    vector<double> dcounts;
    int extrapoints;
    PyObject *pylist;
    int numbins=10;
    
    {
      double min, binsize;
      const double qNaN=numeric_limits<double>::quiet_NaN();
      double defaultMin=qNaN, defaultMax=qNaN;
      vector<double> flist;

      if (   PyArg_ParseTuple(args, "O|idd", &pylist, &numbins, &defaultMin, &defaultMax) 
          && PyList2flist(pylist, flist)) {

        if ((defaultMin!=qNaN) && (defaultMax!=qNaN))
          HISTOSWITCH((flist, counts, min, binsize, extrapoints, defaultMin, defaultMax, numbins),
                      (flist, dcounts, min, binsize, extrapoints, defaultMin, defaultMax, numbins))
        else
          HISTOSWITCH((flist, counts, min, binsize, extrapoints, numbins),
                      (flist, dcounts, min, binsize, extrapoints, numbins))

        return Py_BuildValue("Nddi",
                             function==CALL_RELFREQ ? flist2PyList(dcounts) : ilist2PyList(counts),
                             min, binsize, extrapoints);
      }
    }

    PyErr_Clear();

    {
      PyObject *pyMin=NULL, *pyMax=NULL;
      vector<PyWrapper> wlist;
      PyWrapper min, binsize;

      if (   PyArg_ParseTuple(args, "O|iOO", &pylist, &numbins, &pyMin, &pyMax)
          && PyList2wlist(pylist, wlist)) {

        if (pyMin && pyMax)
          HISTOSWITCH((wlist, counts, min, binsize, extrapoints, PyWrapper(pyMin), PyWrapper(pyMax), numbins),
                      (wlist, dcounts, min, binsize, extrapoints, PyWrapper(pyMin), PyWrapper(pyMax), numbins))
        else
          HISTOSWITCH((wlist, counts, min, binsize, extrapoints, numbins),
                      (wlist, dcounts, min, binsize, extrapoints, numbins))

        return Py_BuildValue("NNNi", 
                             function==CALL_RELFREQ ? flist2PyList(dcounts) : ilist2PyList(counts),
                             (PyObject *)min, (PyObject *)binsize, extrapoints);
      }
    }

    PYERROR(PyExc_TypeError, "histogram: invalid arguments", PYNULL);
  PyCATCH
}


PyObject *py_histogram(PyObject *, PyObject *args)
{ return py_histograms(args, CALL_HISTOGRAM); }

PyObject *py_cumfreq(PyObject *, PyObject *args)
{ return py_histograms(args, CALL_CUMFREQ); }

PyObject *py_relfreq(PyObject *, PyObject *args)
{ return py_histograms(args, CALL_RELFREQ); }



/* *********** VARIABILITY ************/

T_FROM_LIST(samplevar)
T_FROM_LIST(samplestdev)
T_FROM_LIST(var)
T_FROM_LIST(stdev)
T_FROM_LIST(sterr)
T_FROM_LIST_T(z)
LIST_FROM_LIST(zs)


/* *********** TRIMMING FUNCTIONS ************/

LIST_FROM_LIST_DOUBLE(trimboth)

PyObject *py_trim1(PyObject *, PyObject *args) \
{ PyTRY
    PyObject *pylist;
    double mom;
    char *which=NULL;

    if (   PyArg_ParseTuple(args, "Od|s", &pylist, &mom, &which)) {

      bool right;
      if (!which || strcmp(which, "right")==0)
        right=true;
      else if (strcmp(which, "left")==0)
        right=false;
      else
        PYERROR(PyExc_AttributeError, "trim1: invalid 'tail' argument", PYNULL);

      vector<double> flist;
      if (PyList2flist(pylist, flist)) {
        vector<double> fres;
        trim1(flist, mom, fres, right);
        return flist2PyList(fres);
      }

      PyErr_Clear();

      vector<PyWrapper> wlist;
      if (PyList2wlist(pylist, wlist)) {
        vector<PyWrapper> wres;
        trim1(wlist, mom, wres, right);
        return wlist2PyList(wres);
      }
    }

    PYERROR(PyExc_AttributeError, "trim1: invalid arguments", PYNULL);
  PyCATCH
}


/* *********** CORRELATION FUNCTIONS************/

T_T_FROM_LIST_LIST(pearsonr)
DOUBLE_DOUBLE_FROM_LIST_LIST(spearmanr)
DOUBLE_DOUBLE_FROM_LIST_LIST(pointbiserialr)
DOUBLE_DOUBLE_FROM_LIST_LIST(kendalltau)

PyObject *py_linregress(PyObject *, PyObject *args)
{ PyTRY
    #define VARS r, slope, intercepr, probrs, sterrest

    vector<double> x, y;
    if (args22lists(args, x, y)) {
      double VARS;
      linregress(x, y, VARS);
      return Py_BuildValue("ddddd", VARS);
    }
    
    PyErr_Clear();
    
    vector<PyWrapper> wx, wy;
    if (args22wlists(args, wx, wy)) {
      PyWrapper VARS;
      linregress(wx, wy, VARS);
      return Py_BuildValue("NNNNN", (PyObject *)r, (PyObject *)slope, (PyObject *)intercepr, (PyObject *)probrs, (PyObject *)sterrest);
    }

    #undef VARS
    PYERROR(PyExc_AttributeError, "linregress: two lists expected", PYNULL);
  PyCATCH
}



/* *********** INFERENTIAL STATISTICS ************/

T_T_FROM_LIST_T(ttest_1samp)
T_T_FROM_LIST_LIST(ttest_rel)

PyObject *py_ttest_ind(PyObject *, PyObject *args)
{ PyTRY
    vector<double> x, y;
    if (args22listsne(args, x, y)) {
      double res1, res2;
      res1=ttest_ind(x, y, res2);
      return Py_BuildValue("dd", res1, res2);
    }
   
    PyErr_Clear();
   
    vector<PyWrapper> wx, wy;
    if (args22wlistsne(args, wx, wy)) {
      PyWrapper res1, res2;
      res1=ttest_ind(wx, wy, res2);
      return Py_BuildValue("NN", (PyObject *)res1, (PyObject *)res2);
    }
    
    PYERROR(PyExc_AttributeError, "ttest_ind: two lists of equal size expected", PYNULL);
  PyCATCH
}

PyObject *py_chisquare(PyObject *, PyObject *args)
{ PyTRY
    PyObject *pylist1, *pylist2=PYNULL;
    if (PyArg_ParseTuple(args, "O|O", &pylist1, &pylist2)) {
      vector<double> x, y;
      if (   PyList2flist(pylist1, x)
          && (!pylist2 || PyList2flist(pylist2, y))) {
        double res1, res2;
        res1=chisquare(x, pylist2 ? &y : NULL, res2);
        return Py_BuildValue("dd", res1, res2);
      }

      PyErr_Clear();

      vector<PyWrapper> wx, wy;
      if (   PyList2wlist(pylist1, wx)
          && (!pylist2 || PyList2wlist(pylist2, wy))) {
        PyWrapper res1, res2;
        res1=chisquare(wx, pylist2 ? &wy : NULL, res2);
        return Py_BuildValue("NN", (PyObject *)res1, (PyObject *)res2);
      }

    }
    
    PYERROR(PyExc_AttributeError, "chisquare: one or two lists expected", PYNULL);
  PyCATCH
}


PyObject *py_chisquare2d(PyObject *, PyObject *args)
{ PyTRY
    #define VARS prob, cramerV, contingency_coeff

    vector<vector<double> > x;
    if (args2flist2d(args, x)) {
      double chi2, VARS;
      int df;
      chi2=chisquare2d(x, df, VARS);
      return Py_BuildValue("diddd", chi2, df, VARS);
    }
    
    PyErr_Clear();
    
    vector<vector<PyWrapper> > wx;
    if (args2wlist2d(args, wx)) {
      PyWrapper chi2, VARS;
      int df;
      chi2=chisquare2d(wx, df, VARS);
      return Py_BuildValue("NiNNN", (PyObject *)chi2, df, (PyObject *)prob, (PyObject *)cramerV, (PyObject *)contingency_coeff);
    }

    #undef VARS
    PYERROR(PyExc_AttributeError, "chisquare2d: 2d contingency matrix expected", PYNULL);
  PyCATCH
}


PyObject *py_anova_rel(PyObject *, PyObject *args)
{ PyTRY
    vector<vector<double> > x;
    if (args2flist2d(args, x)) {
      double F, prob;
      int df_bt, df_err;
      F = anova_rel(x, df_bt, df_err, prob);
      return Py_BuildValue("diid", F, df_bt, df_err, prob);
    }
    PYERROR(PyExc_AttributeError, "anova_rel: 2d contingency matrix expected", PYNULL);
  PyCATCH
}


PyObject *py_friedmanf(PyObject *, PyObject *args)
{ PyTRY
    vector<vector<double> > x;
    if (args2flist2d(args, x)) {
      double F, prob, chi2;
      int dfnum, dfden;
      F = friedmanf(x, chi2, dfnum, dfden, prob);
      return Py_BuildValue("diidd", F, dfnum, dfden, prob, chi2);
    }
    PYERROR(PyExc_AttributeError, "friedmanf: 2d contingency matrix expected", PYNULL);
  PyCATCH
}


#define WRAPTEST(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    double res, prob; \
\
    vector<double> x, y; \
    if (args22listsne(args, x, y)) { \
      res=name(x, y, prob); \
      return Py_BuildValue("dd", res, prob); \
    } \
    PyErr_Clear(); \
\
    vector<PyWrapper> wx, wy; \
    if (args22wlistsne(args, wx, wy)) { \
      res=name(wx, wy, prob); \
      return Py_BuildValue("dd", res, prob); \
    } \
    PyErr_Clear(); \
\
    PyObject *pylist, *pygroup=NULL, *pycomp=NULL; \
    if (   PyArg_ParseTuple(args, "OOO", &pylist, &pygroup, &pycomp) \
        && PyList2wlist(pylist, wx)) { \
      res=name(wx, prob, IsTrueCallback(pygroup), LessThanCallback(pycomp)); \
      return Py_BuildValue("dd", res, prob);  \
    } \
\
    PYERROR(PyExc_TypeError, #name": two lists or a list with optional group and compare functions expected.", PYNULL); \
  PyCATCH \
}

WRAPTEST(mannwhitneyu)
WRAPTEST(ranksums)

#undef WRAPTEST

DOUBLE_DOUBLE_FROM_LIST_LIST(wilcoxont)


/* *********** PROBABILITY CALCULATIONS ************/

T_FROM_T(gammln)
T_FROM_T_T_T(betai)
T_FROM_T_T_T(betacf)
T_FROM_T(zprob)
T_FROM_T(erf)
T_FROM_T(erfc)
T_FROM_T(erfcc)
T_FROM_T_T(chisqprob)


PyObject *py_fprob(PyObject *, PyObject *args)
{ 
  PyTRY
    int dfnum, dfden;
    double F;
    if (!PyArg_ParseTuple(args, "iid:fprob", &dfnum, &dfden, &F))
      return NULL;
    return PyFloat_FromDouble(fprob(dfnum, dfden, F));
  PyCATCH;
}


/* *********** RANDOM NUMBERS ***************/

T_FROM_T_T(gasdev)

/* *********** SUPPORT FUNCTIONS ************/


T_FROM_LIST_optT(sum)
LIST_FROM_LIST_optT(cumsum)
T_FROM_LIST_optT(ss)
T_FROM_LIST_LIST_optT(summult)
T_FROM_LIST_optT(sumsquared)
T_FROM_LIST_LIST_optT(sumdiffsquared)


PyObject *py_shellsort(PyObject *, PyObject *args)
{ PyTRY
    vector<double> flist;
    if (args2flist(args, flist)) {
      vector<int> indices;
      vector<double> items;
      if (!shellsort(flist, indices, items))
        PYERROR(PyExc_AttributeError, "shellsort failed", NULL);

      PyObject *pyind=ilist2PyList(indices), *pyitems=flist2PyList(items);
      return Py_BuildValue("NN", pyitems, pyind);
    }

    PyErr_Clear();

    PyObject *pylist, *pycomp=NULL;
    if (   !PyArg_ParseTuple(args, "O|O", &pylist, &pycomp)
        || !PyList_Check(pylist))
      PYERROR(PyExc_AttributeError, "list and optional compare function expected", false)

    vector<PyWrapper> wlist;
    if (PyList2wlist(pylist, wlist)) {
      vector<int> indices;
      vector<PyWrapper> items;

      if (pycomp ? !shellsort(wlist, indices, items, LessThanCallback(pycomp))
                 : !shellsort(wlist, indices, items))
          PYERROR(PyExc_AttributeError, "shellsort failed", NULL);

      PyObject *pyind=ilist2PyList(indices), *pyitems=wlist2PyList(items);
      return Py_BuildValue("NN", pyitems, pyind);
    }
  
    return PYNULL;
  PyCATCH
}


PyObject *py_rankdata(PyObject *, PyObject *args)
{ PyTRY
    vector<double> flist, ranks;
    if (args2flist(args, flist)) {
      if (!rankdata(flist, ranks))
        PYERROR(PyExc_SystemError, "rankdata: failed", NULL);
      return flist2PyList(ranks);
    }
    
    PyErr_Clear();
    
    vector<PyWrapper> wlist;
    PyObject *pylist, *pycomp=NULL;
    if (   !PyArg_ParseTuple(args, "O|O", &pylist, &pycomp)
        || !PyList_Check(pylist))
      PYERROR(PyExc_AttributeError, "rankdata: list and optional compare function expected", PYNULL)

    if (pycomp ? !rankdata(wlist, ranks, LessThanCallback(pycomp))
               : !rankdata(wlist, ranks))
        PYERROR(PyExc_SystemError, "rankdata: failed", NULL);
      return flist2PyList(ranks);

    return PYNULL;
  PyCATCH \
}


/* *********** LOESS ******************************/



int cc_list(PyObject *pylist, void *l)
{ 
  if (!PyList_Check(pylist))
    return 0;

  vector<double> *lst = (vector<double> *)l;

  int len = PyList_Size(pylist);
  *lst = vector<double>();
  lst->reserve(len);

  for(int i = 0; i<len; i++) {
    PyObject *asnum = PyNumber_Float(PyList_GET_ITEM(pylist, i));
    if (!asnum)
      return 0;
    lst->push_back(PyFloat_AsDouble(asnum));
    Py_DECREF(asnum);
  }

  return 1;
}


PyObject *list2python(const vector<double> &lst)
{
  PyObject *res = PyList_New(lst.size());
  int i = 0;
  for(vector<double>::const_iterator li(lst.begin()), le(lst.end()); li!=le; li++, i++)
    PyList_SetItem(res, i, PyFloat_FromDouble(*li));
  return res;
}

typedef void TSampFunc(const vector<double> &, int, vector<double> &);
TSampFunc *sampFuncs[] = {samplingMinimal, samplingFactor, samplingFixed, samplingUniform};


bool getSmootherPars(PyObject *args, vector<pair<double, double> > &points, vector<double> &sampPoints,
                  float &smoothPar, const char *method)
{
  PyObject *pypoints;
  int nPoints;
  int distMethod;
  char buf[20];
  vector<double> xpoints;

  points.clear();
  sampPoints.clear();

 
  if (PyList_Check(PyTuple_GET_ITEM(args, 1))) {
    snprintf(buf, 19, "OO&f:%s", method);
    if (!PyArg_ParseTuple(args, buf, &pypoints, cc_list, &sampPoints, &smoothPar))
      return false;
    distMethod = -1;
  }
  else {
    snprintf(buf, 19, "Oif|i:%s", method);
    if (!PyArg_ParseTuple(args, "Oif|i:loess", &pypoints, &nPoints, &smoothPar, &distMethod))
      return false;
    if ((distMethod < DISTRIBUTE_MINIMAL) || (distMethod > DISTRIBUTE_UNIFORM))
      PYERROR(PyExc_TypeError, "invalid point distribution method", false);
  }


  PyObject *iter = PyObject_GetIter(pypoints);
  if (!iter)
    PYERROR(PyExc_TypeError, "a list (or a tuple) of points expected", false);

  PyObject *item;
  for (int i = 0; (item = PyIter_Next(iter)) != NULL; i++) {
    PyObject *pyx = NULL, *pyy = NULL;
    if (   !PyTuple_Check(item)
        || (PyTuple_Size(item)!=2)
        || ((pyx = PyNumber_Float(PyTuple_GetItem(item, 0))) == NULL)
        || ((pyy = PyNumber_Float(PyTuple_GetItem(item, 1))) == NULL)) {
      Py_XDECREF(pyx);
      Py_DECREF(item);
      Py_DECREF(iter);
      PyErr_Format(PyExc_TypeError, "invalid point at index %i", i);
      return false;
    }

    points.push_back(pair<double, double>(PyFloat_AsDouble(pyx), PyFloat_AsDouble(pyy)));
    if (distMethod != -1)
      xpoints.push_back(PyFloat_AsDouble(pyx));

    Py_DECREF(pyy);
    Py_DECREF(pyx);
    Py_DECREF(item);
  }
  Py_DECREF(iter);
  
  if (distMethod != -1)
    sampFuncs[distMethod](xpoints, nPoints, sampPoints);

  return true;
}


PyObject *curve2PyCurve(const vector<double> xs, const vector<pair<double, double> > yvars)
{
  PyObject *pypoints = PyList_New(xs.size());
  int i = 0;
  vector<double>::const_iterator xi(xs.begin());
  vector<pair<double, double> >::const_iterator yvi(yvars.begin()), yve(yvars.end());
  for (; yvi != yve; yvi++, xi++)
    PyList_SetItem(pypoints, i++, Py_BuildValue("fff", *xi, (*yvi).first, (*yvi).second));
  return pypoints;
}


PyObject *py_loess(PyObject *, PyObject *args) 
{ PyTRY
    float windowProp;
    vector<pair<double, double> > points;
    vector<double> sampPoints;

    if (!getSmootherPars(args, points, sampPoints, windowProp, "loess"))
      return PYNULL;

    vector<pair<double, double> > loess_curve;
    loess(sampPoints, points, windowProp, loess_curve);

    return curve2PyCurve(sampPoints, loess_curve);
  PyCATCH
}


PyObject *py_lwr(PyObject *, PyObject *args) 
{ PyTRY
    float sigmaPercentile;
    vector<pair<double, double> > points;
    vector<double> sampPoints;

    if (!getSmootherPars(args, points, sampPoints, sigmaPercentile, "lwr"))
      return PYNULL;

    vector<pair<double, double> > lwr_curve;
    lwr(sampPoints, points, sigmaPercentile, lwr_curve);

    return curve2PyCurve(sampPoints, lwr_curve);
  PyCATCH
}


/* *********** COMBINATORIAL FUNCTIONS ************/


#include "lcomb.hpp"

#define DOUBLE_FROM_INT(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    int i; \
    if (!PyArg_ParseTuple(args, "i", &i)) \
      PYERROR(PyExc_AttributeError, "integer expected", PYNULL) \
    double res=name(i); \
    return Py_BuildValue("d", res); \
  PyCATCH \
}

#define DOUBLE_FROM_INT_INT(name) \
PyObject *py_##name(PyObject *, PyObject *args) \
{ PyTRY \
    int i1, i2; \
    if (!PyArg_ParseTuple(args, "ii", &i1, &i2)) \
      PYERROR(PyExc_AttributeError, "integer expected", PYNULL) \
    double res=name(i1, i2); \
    return Py_BuildValue("d", res); \
  PyCATCH \
}

DOUBLE_FROM_INT(fact)
DOUBLE_FROM_INT_INT(comb)
DOUBLE_FROM_INT_INT(stirling2)
DOUBLE_FROM_INT(bell)

DOUBLE_FROM_INT(logfact)
DOUBLE_FROM_INT_INT(logcomb)


#define PY_SAMPLING(name) \
PyObject *py_sampling##name(PyObject *, PyObject *args) \
{ \
  vector<double> points; \
  int nPoints; \
  \
  if (!PyArg_ParseTuple(args, "O&i:sampling" #name, cc_list, &points, &nPoints)) \
    return NULL; \
  \
  vector<double> lst; \
  sampling##name(points, nPoints, lst); \
  return list2python(lst); \
}

PY_SAMPLING(Factor)
PY_SAMPLING(Fixed)
PY_SAMPLING(Uniform)
PY_SAMPLING(Minimal)

#undef PY_SAMPLING

/* *********** EXPORT DECLARATIONS ************/

#define DECLARE(name) \
 {#name, (binaryfunc)py_##name, METH_VARARGS},

PyMethodDef statc_functions[]={
     DECLARE(geometricmean)
     DECLARE(harmonicmean)
     DECLARE(mean)
     DECLARE(median)
     {"medianscore", (binaryfunc)py_median, METH_VARARGS},
     DECLARE(mode)

     DECLARE(moment)
     DECLARE(variation)
     DECLARE(skewness)
     DECLARE(kurtosis)

     DECLARE(scoreatpercentile)
     DECLARE(percentileofscore)
     DECLARE(histogram)
     DECLARE(cumfreq)
     DECLARE(relfreq)


     DECLARE(samplevar)
     {"samplestd", (binaryfunc)py_samplestdev, METH_VARARGS},
     DECLARE(var)
     {"std", (binaryfunc)py_stdev, METH_VARARGS},
     DECLARE(z)
     DECLARE(zs)
     DECLARE(sterr)

     DECLARE(trimboth)
     DECLARE(trim1)

     DECLARE(pearsonr)
     DECLARE(spearmanr)
     DECLARE(pointbiserialr)
     DECLARE(kendalltau)
     DECLARE(linregress)

     DECLARE(ttest_1samp)
     DECLARE(ttest_ind)
     DECLARE(ttest_rel)
     DECLARE(chisquare)
     DECLARE(chisquare2d)
     DECLARE(anova_rel)
     DECLARE(friedmanf)
     DECLARE(mannwhitneyu)
     DECLARE(ranksums)
     DECLARE(wilcoxont)

     DECLARE(chisqprob)
     DECLARE(zprob)
     DECLARE(fprob)
     DECLARE(betacf)
     DECLARE(betai)
     DECLARE(erf)
     DECLARE(erfc)
     DECLARE(erfcc)
     DECLARE(gammln)


     DECLARE(sum)
     DECLARE(ss)
     DECLARE(sumsquared)
     DECLARE(summult)
     DECLARE(cumsum)
     DECLARE(sumdiffsquared)
     DECLARE(shellsort)
     DECLARE(rankdata)
     DECLARE(spearmanr)

     DECLARE(gasdev)

     DECLARE(fact)
     DECLARE(comb)
     DECLARE(stirling2)
     DECLARE(bell)

     DECLARE(logfact)
     DECLARE(logcomb)

     {"loess", (binaryfunc)py_loess, METH_VARARGS},
     {"lwr", (binaryfunc)py_lwr, METH_VARARGS},
     DECLARE(samplingFactor)
     DECLARE(samplingFixed)
     DECLARE(samplingMinimal)
     DECLARE(samplingUniform)

     {NULL, NULL}
};

#undef T_FROM_LIST
#undef T_FROM_LIST_optT
#undef T_FROM_LIST_LIST
#undef T_FROM_LIST_LIST_optT
#undef LIST_FROM_LIST
#undef LIST_FROM_LIST_optT
#undef T_FROM_LIST_T
#undef T_FROM_LIST_plus
#undef T_FROM_LIST_INT
#undef T_FROM_LIST_DOUBLE
#undef LIST_FROM_LIST_plus
#undef LIST_FROM_LIST_INT
#undef LIST_FROM_LIST_DOUBLE
#undef DOUBLE_DOUBLE_FROM_LIST_LIST
#undef T_T_FROM_LIST_LIST
#undef T_T_FROM_LIST_plus
#undef T_T_FROM_LIST
#undef T_T_FROM_LIST_INT
#undef T_T_FROM_LIST_DOUBLE
#undef DOUBLE_DOUBLE_FROM_LIST_plus
#undef DOUBLE_DOUBLE_FROM_LIST_INT
#undef DOUBLE_DOUBLE_FROM_LIST_DOUBLE
#undef T_T_FROM_LIST_T
#undef T_FROM_T
#undef T_FROM_T_T_T
#undef CALL_HISTOGRAM
#undef CALL_CUMFREQ
#undef CALL_RELFREQ
#undef HISTOSWITCH
#undef WRAPTEST
#undef DECLARE

#undef PyTRY
#undef PyCATCH
#undef PYNULL
