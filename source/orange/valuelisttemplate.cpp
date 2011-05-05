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
  #pragma warning (disable : 4786 4114 4018 4267 4244 4702 4710 4290)
#endif


#include "cls_orange.hpp"
#include "vectortemplates.hpp"
#include "cls_value.hpp"
#include "valuelisttemplate.hpp"


int ccn_func_Variable(PyObject *, void *);

PyObject *TValueListMethods::_CreateEmptyList(PyTypeObject *type, PVariable var)
{ return WrapNewOrange(mlnew TValueList(var), type); }


PValueList TValueListMethods::P_FromArguments(PyObject *arg, PVariable var)
{ if (!PySequence_Check(arg)) {
    PyErr_Format(PyExc_TypeError, "invalid arguments for 'ValueList' constructor (sequence expected)");
    return PValueList();
  }

  PValueList aList = mlnew TValueList(var);
  for(Py_ssize_t i=0, e=PySequence_Size(arg); i!=e; i++) {
    PyObject *pyobj=PySequence_GetItem(arg, i);
    TValue item;
    bool ok = convertFromPython(pyobj, item, var);
    if (!ok) {
      PyErr_Format(PyExc_TypeError, "element at index %i is of wrong type ('%s')", i, pyobj ? pyobj->ob_type->tp_name : "None");
      Py_DECREF(pyobj);
      return PValueList();
    }
    Py_DECREF(pyobj);
    aList->push_back(item);
  }

  return aList;
}


PyObject *TValueListMethods::_FromArguments(PyTypeObject *type, PyObject *arg, PVariable var)
{ PValueList newList = P_FromArguments(arg, var);
  return newList ? WrapOrange(newList) : PYNULL;
};


PyObject *TValueListMethods::_new(PyTypeObject *type, PyObject *args, PyObject *)
{ if (!args || (PySequence_Check(args) && !PySequence_Size(args)))
    return _CreateEmptyList(type);

  PyObject *arg;
  PVariable var;
  if (   PyArg_ParseTuple(args, "O|O&", &arg, ccn_func_Variable, &var)
      && PySequence_Check(arg))
    return _FromArguments(type, arg, var);

  return _FromArguments(type, args, var);
}


PyObject *TValueListMethods::_getitem(TPyOrange *self, Py_ssize_t index)
{ PyTRY
    CAST_TO(TValueList, aList)
    return checkIndex(index, aList->size()) ? Value_FromVariableValue(aList->variable, aList->operator[](index)) : PYNULL;
  PyCATCH
}


int TValueListMethods::_setitem(TPyOrange *self, Py_ssize_t index, PyObject *item)
{ PyTRY
    CAST_TO_err(TValueList, aList, -1)
    if (!checkIndex(index, aList->size()))
      return -1;
    if (item==NULL) {
      aList->erase(aList->begin()+index);
    }
    else {
      TValue citem;
      if (!convertFromPython(item, citem, aList->variable))
        return -1;
      aList->operator[](index)=citem;
    }

    return 0;
  PyCATCH_1
}


int TValueListMethods::_cmp(TPyOrange *self, PyObject *other)
{ PyTRY
    PyObject *myItem = NULL, *hisItem = NULL;
    try {
      PyObject *iterator = PyObject_GetIter(other);
      if (!iterator) {
        PyErr_Format(PyExc_TypeError, "'%s.__cmp__': not a sequence", self->ob_type->tp_name);
        return -1;
      }

      CAST_TO_err(TValueList, aList, -1)
      int result;
      for(TValueList::iterator ii(aList->begin()), ei(aList->end()); ii!=ei; ii++) {
        myItem = Value_FromVariableValue(aList->variable, *ii);
        hisItem = PyIter_Next(iterator);
        if (!hisItem) {
          Py_DECREF(myItem);
          Py_DECREF(iterator);
          return PyErr_Occurred() ? -1 : 1;
        }

        int err = PyObject_Cmp(myItem, hisItem, &result);
        Py_DECREF(myItem);
        Py_DECREF(hisItem);
        myItem = NULL;
        hisItem = NULL;

        if (err == -1) {
          Py_DECREF(iterator);
          return -1;
        }
        else
          if (result!=0) {
            Py_DECREF(iterator);
            return result;
          }
      }

      hisItem = PyIter_Next(iterator);
      Py_DECREF(iterator);

      if (!hisItem)
        return PyErr_Occurred() ? -1 : 0;

      Py_DECREF(hisItem);
      return -1;
    }
    catch (exception err) {
      Py_XDECREF(myItem);
      Py_XDECREF(hisItem);
      throw;
    }
  PyCATCH_1
}


PyObject *TValueListMethods::_str(TPyOrange *self)
{
  PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
  if (result)
    return result;

  CAST_TO(TValueList, aList);
  string res("<");
  for(const_iterator bi(aList->begin()), ei(bi), ee(aList->end()); ei!=ee; ei++) {
    if (ei!=bi)
      res += ", ";

    PyObject *obj = Value_FromVariableValue(aList->variable, *ei);
    PyObject *repred = PyObject_Str(obj);
    res += PyString_AsString(repred);
    Py_DECREF(obj);
    Py_DECREF(repred);
  }
  res += ">";
  return PyString_FromString(res.c_str());
}


PyObject *TValueListMethods::_append(TPyOrange *self, PyObject *item)
{ PyTRY
    CAST_TO(TValueList, aList);

    TValue obj;
    if (!convertFromPython(item, obj, aList->variable))
      return PYNULL;

    aList->push_back(obj);
    RETURN_NONE;
  PyCATCH
}

PyObject *TValueListMethods::_count(TPyOrange *self, PyObject *item)
{ PyTRY
    CAST_TO(TValueList, aList);

    TValue obj;
    if (!convertFromPython(item, obj, aList->variable))
      return PYNULL;

    int cnt=0;
    for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
      if (obj==*bi)
        cnt++;
    return PyInt_FromLong(cnt);
  PyCATCH
}


int TValueListMethods::_contains(TPyOrange *self, PyObject *item)
{ PyTRY
    CAST_TO_err(TValueList, aList, -1);

    TValue obj;
    if (!convertFromPython(item, obj, aList->variable))
      return -1;

    for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
      if (obj==*bi)
        return 1;
    return 0;
  PyCATCH_1
}


PyObject *TValueListMethods::_filter(TPyOrange *self, PyObject *args)
{
  PyTRY
    PyObject *filtfunc=NULL;
    if (!PyArg_ParseTuple(args, "|O:filter", &filtfunc))
      return PYNULL;

    PyObject *emtuple = PyTuple_New(0);
    PyObject *emdict = PyDict_New();
    PyObject *newList = self->ob_type->tp_new(self->ob_type, emtuple, emdict);
    Py_DECREF(emtuple);
    Py_DECREF(emdict);
    emtuple = NULL;
    emdict = NULL;
    if (!newList)
      return NULL;

    CAST_TO(TValueList, aList)
    NAME_CAST_TO(TValueList, newList, cList)
    for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++) {
      PyObject *lel=Value_FromVariableValue(aList->variable, *bi);
      if (filtfunc) {
        PyObject *filtres=PyObject_CallFunction(filtfunc, "O", lel);
        Py_DECREF(lel);
        if (!filtres)
          throw pyexception();
        lel=filtres;
      }
      if (PyObject_IsTrue(lel))
        cList->push_back(*bi);
      Py_DECREF(lel);
    }

    return newList;
  PyCATCH;
}


PyObject *TValueListMethods::_index(TPyOrange *self, PyObject *item)
{ PyTRY
    CAST_TO(TValueList, aList);

    TValue obj;
    if (!convertFromPython(item, obj, aList->variable))
      return PYNULL;

    for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
      if (obj==*bi)
        return PyInt_FromLong(bi-aList->begin());
    PYERROR(PyExc_ValueError, "list.index(x): x not in list", PYNULL)
  PyCATCH
}


PyObject *TValueListMethods::_insert(TPyOrange *self, PyObject *args)
{ PyTRY
    CAST_TO(TValueList, aList);

    PyObject *obj;
    int index;
    TValue item;

    if (!PyArg_ParseTuple(args, "iO", &index, &obj))
    	return PYNULL;

    Py_ssize_t sindex = index;
    if (   !checkIndex(sindex, aList->size())
        || !convertFromPython(obj, item, aList->variable))
      return PYNULL;

    aList->insert(aList->begin()+sindex, item);
    RETURN_NONE;
  PyCATCH
}


PyObject *TValueListMethods::_native(TPyOrange *self)
{ PyTRY
    CAST_TO(TValueList, aList);

    PyObject *newList = PyList_New(aList->size());

    Py_ssize_t i=0;
    for(const_iterator li = aList->begin(), le = aList->end(); li!=le; li++)
      PyList_SetItem(newList, i++, Value_FromVariableValue(aList->variable, *li));

    return newList;
  PyCATCH
}


PyObject *TValueListMethods::_pop(TPyOrange *self, PyObject *args)
{ PyTRY
    CAST_TO(TValueList, aList);
    int idx=aList->size()-1;
    if (!PyArg_ParseTuple(args, "|i:pop", &idx))
      return PYNULL;

    PyObject *ret = _getitem(self, idx);
    if (!ret)
      return PYNULL;

    aList->erase(aList->begin()+idx);
    return ret;
  PyCATCH
}


PyObject *TValueListMethods::_remove(TPyOrange *self, PyObject *item)
{ PyTRY
    CAST_TO(TValueList, aList);

    TValue obj;
    if (!convertFromPython(item, obj, aList->variable))
      return PYNULL;

    for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
      if (obj==*bi) {
        aList->erase(bi);
        RETURN_NONE;
      }
    PYERROR(PyExc_ValueError, "remove(x): x not in list", PYNULL)
  PyCATCH
}


TValueListMethods::TCmpByCallback::TCmpByCallback(PVariable var, PyObject *func)
: variable(var)
{ if (!PyCallable_Check(func))
    raiseErrorWho("CmpByCallback", "compare object not callable");

  cmpfunc=func;
  Py_INCREF(cmpfunc);
}

TValueListMethods::TCmpByCallback::TCmpByCallback(const TCmpByCallback &other)
  : cmpfunc(other.cmpfunc)
{ Py_INCREF(cmpfunc); }

TValueListMethods::TCmpByCallback::~TCmpByCallback()
{ Py_DECREF(cmpfunc);
}

bool TValueListMethods::TCmpByCallback::operator()(const TValue &x, const TValue &y) const
{ if (cmpfunc) {
    PyObject *pyx=Value_FromVariableValue(variable, x), *pyy=Value_FromVariableValue(variable, y);
    PyObject *cmpres=PyObject_CallFunction(cmpfunc, "OO", pyx, pyy);
    Py_DECREF(pyx);
    Py_DECREF(pyy);

    if (!cmpres)
      throw pyexception();

    int res=PyInt_AsLong(cmpres);
    Py_DECREF(cmpres);

    return res<0;
  }
  else
    return x.compare(y)==-1;
}

PyObject *TValueListMethods::_sort(TPyOrange *self, PyObject *args)
{
  PyTRY
    PyObject *cmpfunc=NULL;
    if (!PyArg_ParseTuple(args, "|O:sort", &cmpfunc))
      return PYNULL;

    CAST_TO(TValueList, aList)
    std::sort(aList->begin(), aList->end(), TCmpByCallback(aList->variable, cmpfunc));

    RETURN_NONE;
  PyCATCH;
}

