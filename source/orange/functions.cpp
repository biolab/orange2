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
 #pragma warning (disable : 4786 4114 4018 4267)
#endif

#include <stdlib.h>
#include <string>

#include "random.hpp"

#include "vars.hpp"
#include "meta.hpp"
#include "tabdelim.hpp"
#include "c45inter.hpp"
#include "table.hpp"
#include "filter.hpp"
#include "spec_gen.hpp"
#include "Python.h"

#include "cls_orange.hpp"
//#include "mlpy.hpp"

using namespace std;

bool convertFromPython(PyObject *, float &);


PyObject *newmetaid(PyObject *, PyObject *) PYARGS(0,"() -> int")
{ PyTRY
    return PyInt_FromLong(getMetaID()); 
  PyCATCH
}

PyObject *setWarningLevel(PyObject *, PyObject *arg) PYARGS(METH_O, "(bool) -> None")
{ 
  exhaustiveWarnings = (PyObject_IsTrue(arg) != 0);
  RETURN_NONE;
}


void registerVariableType(PyObject *variable);

PyObject *registerPythonVariable(PyObject *, PyObject *vartype) PYARGS(METH_O, "(class) -> None")
{
  PyTRY
    registerVariableType(vartype);
    RETURN_NONE;
  PyCATCH
}


PyObject *setoutput(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, format-name, function) -> None")
{ PyTypeObject *type;
  char *formatname;
  PyObject *function;
  if (!PyArg_ParseTuple(args, "OsO", (PyObject **)&type, &formatname, &function))
    return PYNULL;

  if (!PyType_IsSubtype(type, (PyTypeObject *)&PyOrOrange_Type))
    PYERROR(PyExc_TypeError, "Orange or a subclass type expected", PYNULL);

  char os[256] = "__output_";

  PyObject *newmethod = PyMethod_New(function, NULL, (PyObject *)type);
  if (!newmethod)
    PYERROR(PyExc_TypeError, "invalid output function", PYNULL);

  PyDict_SetItemString(type->tp_dict, strcat(os, formatname), newmethod);
  Py_DECREF(newmethod);

  RETURN_NONE;
}


PyObject *removeoutput(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, format-name) -> None")
{ PyTypeObject *type;
  char *formatname;
  if (!PyArg_ParseTuple(args, "Os", (PyObject **)&type, &formatname))
    return PYNULL;

  if (!PyType_IsSubtype(type, (PyTypeObject *)&PyOrOrange_Type))
    PYERROR(PyExc_TypeError, "Orange or a subclass type expected", PYNULL);

  char os[256] = "__output_";
  strcat(os, formatname);

  if (!PyDict_GetItemString(type->tp_dict, os)) {
    PyErr_Format(PyExc_TypeError, "'%s' has no output '%s'", type->tp_name, formatname);
    return PYNULL;
  }
  
  PyDict_DelItemString(type->tp_dict, os);
  RETURN_NONE;
}



PyObject *__addmethod(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, name, function) -> None")
{ PyTypeObject *type;
  char *name;
  PyObject *function;
  if (!PyArg_ParseTuple(args, "OsO", (PyObject **)&type, &name, &function))
    return PYNULL;

  if (!PyType_IsSubtype(type, (PyTypeObject *)&PyOrOrange_Type))
    PYERROR(PyExc_TypeError, "Orange or a subclass type expected", PYNULL);

  PyDict_SetItemString(type->tp_dict, name, PyMethod_New(function, NULL, (PyObject *)type));
  RETURN_NONE;
}


PyObject *__removemethod(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(type, name) -> None")
{ PyTypeObject *type;
  char *name;
  if (!PyArg_ParseTuple(args, "Os", (PyObject **)&type, &name))
    return PYNULL;

  if (!PyType_IsSubtype(type, (PyTypeObject *)&PyOrOrange_Type))
    PYERROR(PyExc_TypeError, "Orange or a subclass type expected", PYNULL);

  if (!PyDict_GetItemString(type->tp_dict, name)) {
    PyErr_Format(PyExc_TypeError, "'%s' has no method '%s'", type->tp_name, name);
    return PYNULL;
  }
  
  PyDict_DelItemString(type->tp_dict, name);
  RETURN_NONE;
}






PyObject *select(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(selector, sequence) -> selected-sequence")
{ PyObject *selector, *sequence;
  if (   !PyArg_ParseTuple(args, "OO", &selector, &sequence)
      || !PyList_Check(selector)
      || !PyList_Check(sequence)
      || (PyList_Size(selector)!=PyList_Size(sequence)))
    PYERROR(PyExc_TypeError, "select requires two lists of equal sizes", PYNULL)

  int size=PyList_Size(selector);
  PyObject *result=PyList_New(0);
  for(int i=0; i<size; i++)
    if (PyObject_IsTrue(PyList_GetItem(selector, i)))
      PyList_Append(result, PyList_GetItem(sequence, i));

  return result;
}



PyObject *frange(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "([[start=step], stop=1.0+step], step) -> [start+i*step]")
{ double start=0.0, stop=1.0, step;
  if (PyArg_ParseTuple(args, "d", &step))
    start=step;
  else {
    PyErr_Clear();
    if (PyArg_ParseTuple(args, "dd", &stop, &step))
      start=step;
    else {
      PyErr_Clear();
      if (!PyArg_ParseTuple(args, "ddd", &start, &stop, &step))
        PYERROR(PyExc_AttributeError, "1-3 arguments expected", PYNULL);
    }
  }

  PyObject *pylist=PyList_New(0);
  int i;
  double f;
  for(i=0, f=start, stop+=1e-10; f<stop; f=start + (++i)*step) {
    PyObject *nr=PyFloat_FromDouble(f);
    PyList_Append(pylist, nr);
    Py_DECREF(nr);
  }
  return pylist;
}
  

PyObject *compiletime(PyObject *, PyObject *) PYARGS(METH_NOARGS, "() -> time")
{ tm time;

  char *compdate= __DATE__;
  compdate[3]=compdate[6]=char(0);

  char *months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

  time.tm_year=atoi(compdate+7)-1900;
  time.tm_mday=atoi(compdate+4);
  for(time.tm_mon=0; (time.tm_mon<12) && strcmp(months[time.tm_mon], compdate); time.tm_mon++);
  if (time.tm_mon==12) time.tm_mon=0;

  char *comptime= __TIME__;
  comptime[2]=comptime[5]=0;
  time.tm_hour=atoi(comptime);
  time.tm_min=atoi(comptime+3);
  time.tm_sec=atoi(comptime+6);
  time.tm_isdst=0;

  time_t tt=mktime(&time);
  tm *ntime=localtime(&tt);
  
  time.tm_mon+=1;
  time.tm_year+=1900;

  return Py_BuildValue("iiiiiiiii", time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec, ntime->tm_wday, ntime->tm_hour, ntime->tm_yday, ntime->tm_isdst);
}


int pt_ExampleGenerator(PyObject *args, void *egen);

PyObject *__arrayDistance(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(table1, table2) -> distance")
{
  PExampleGenerator g1, g2;
  if (!PyArg_ParseTuple(args, "O&O&:__arrayDistance", pt_ExampleGenerator, &g1, pt_ExampleGenerator, &g2))
    return PYNULL;

  const int nattrs = g1->domain->attributes->size();
  if ((g2->domain->attributes->size() != nattrs) || (g1->numberOfExamples() != g2->numberOfExamples()))
    PYERROR(PyExc_AttributeError, "two example tables with equal number of attributes and examples expected", PYNULL);

  if (g1->domain->classVar || g2->domain->classVar)
    raiseWarning(false, "__arrayDistance ignores class values");

  TVarList::const_iterator vi, ve;
  for(vi = g1->domain->attributes->begin(), ve = g1->domain->attributes->end(); vi!=ve; vi++)
    if ((*vi)->varType != TValue::FLOATVAR) {
      PyErr_Format(PyExc_TypeError, "attribute %s is not continuous", (*vi)->name.c_str());
      return PYNULL;
    }
  if (g1->domain != g2->domain)
  for(vi = g2->domain->attributes->begin(), ve = g2->domain->attributes->end(); vi!=ve; vi++)
    if ((*vi)->varType != TValue::FLOATVAR) {
      PyErr_Format(PyExc_TypeError, "attribute %s is not continuous", (*vi)->name.c_str());
      return PYNULL;
    }

  float sum = 0.0;
  int existing = 0;
  for(TExampleIterator g1i(g1->begin()), g2i(g2->begin()); g1i; ++g1i, ++g2i) {
    TExample::const_iterator e1i((*g1i).begin()), e2i((*g2i).begin());
    for(int i = nattrs; i--; e1i++, e2i++)
      if (!(*e1i).isSpecial() && !(*e2i).isSpecial()) {
        const float d = (*e1i).floatV - (*e2i).floatV;
        sum += d*d;
        existing++;
      }
  }

  if (!existing)
    PYERROR(PyExc_AttributeError, "no defined values", PYNULL);

  return PyFloat_FromDouble(sqrt(sum/existing));
}


/********** OBSOLETE ***************/

PyObject *setrandseed(PyObject *, PyObject *args) PYARGS(METH_VARARGS, "(int seed) -> None")
{ int rs;
  if (!PyArg_ParseTuple(args, "i", &rs))
    PYERROR(PyExc_TypeError, "integer parameter expected", PYNULL);

  srand(rs);
  RETURN_NONE;
}

/* This is to trick makedep.py
#include "functions.px"
*/

