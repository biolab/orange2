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


/* 

How to export a vector to Python?

a) If vector is a property of some object (probably inherited from TOrangeVector
or _TOrangeVector), you only need to define it as properties.
For instance, 'Domain' returns 'attributes' as 'VarList' only by defining it
as property_wr.

b) If the class behaves vector-like by using VECTOR_INTERFACE itself, you don't
need to do this. (But you should not forget to write traverse and dropReferences
if the vector contains wrapped oranges!)


This header is needed to define methods of the exported vectors.
Each vector type is exported as a separate type in Python, and they are in no
hereditary relation.

You should open defvectors.py and add the vector to one of its lists, either
one for vectors of wrapped or of unwrapped elements. The difference between
a) and b) is only in list names (the first uses generic names like VarList or
ClassifierList, the other can have names like Preprocess or Filter_sameValue,
although they vector_interface vectors of Preprocessors or ValueRanges).

Run the script.

If the vector contains wrapped elements, find the defined methods in
lib_vectors_auto.txt and copy&paste the methods to a corresponding lib_*.cpp.
You can remove are manually redefine some of them if you need. (But keep notes
of this!).

If the vector contains unwrapped elements, the corresponding methods will be added
to lib_vectors.cpp (this file is generated entirely by defvector.py and you should
not change it). You only need to make sure there are corresponding
convert[(To)|(From)]Python methods defined somewhere (preferably in c2py.hpp).

In any case, run pyxtract.py to include the vector types in corresponding .px files.
*/

#ifndef __VECTORTEMPLATES_HPP
#define __VECTORTEMPLATES_HPP

/********************************

This file includes constructors and specialized methods for Orange vectors

*********************************/

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "orvector.hpp"
#include "c2py.hpp"


inline bool checkIndex(Py_ssize_t &index, Py_ssize_t max)
{ if (index<0)
    index += max;
  if ((index<0) || (index>=max)) {
    PyErr_Format(PyExc_IndexError, "index %i out of range 0-%i", index, max-1);
    return false;
  }
  return true;
}


inline bool checkIndices(Py_ssize_t &start, Py_ssize_t &stop, Py_ssize_t max)
{ if (stop>max)
    stop=max;

  if (start>stop) {
    PyErr_Format(PyExc_IndexError, "invalid indices for slice");
    return false;
  }
  return true;
}


/* This contains methods that are same for lists of wrapped and of unwrapped items */
template<class _WrappedListType, class _ListType>
class CommonListMethods {
public:
  typedef typename _ListType::iterator iterator;
  typedef typename _ListType::const_iterator const_iterator;

  static PyObject *_CreateEmptyList(PyTypeObject *type)
  { return WrapNewOrange(mlnew _ListType(), type);
  }

  static Py_ssize_t _len(TPyOrange *self)
  { PyTRY
      CAST_TO_err(_ListType, aList, -1);
      return aList->size();
    PyCATCH_1
  }  

  static PyObject *_repeat(TPyOrange *self, Py_ssize_t times)
  { PyObject *emtuple = NULL, *emdict = NULL, *newList = NULL;
    try {
      emtuple = PyTuple_New(0);
      emdict = PyDict_New();
      newList = self->ob_type->tp_new(self->ob_type, emtuple, emdict);
      Py_DECREF(emtuple); 
      emtuple = NULL;
      Py_DECREF(emdict);
      emdict = NULL;
      if (!newList)
        return NULL;

      CAST_TO(_ListType, aList)
      NAME_CAST_TO(_ListType, newList, cList)
      while(times-- >0)
        for (const_iterator li = aList->begin(), le = aList->end(); li!=le; li++)
          cList->push_back(*li);

      return newList;
    }
      
    catch (exception err)
      { Py_XDECREF(emtuple);
        Py_XDECREF(emdict);
        Py_XDECREF(newList);
        PYERROR(PyExc_Exception, err.what(), PYNULL);
      }
  }

  static PyObject *_getslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop)
  { PyObject *emtuple = NULL, *emdict = NULL, *newList = NULL;
    try {
      CAST_TO(_ListType, aList)
      if (!checkIndices(start, stop, aList->size()))
        return NULL;

      emtuple = PyTuple_New(0);
      emdict = PyDict_New();
      newList = self->ob_type->tp_new(self->ob_type, emtuple, emdict);
      Py_DECREF(emtuple); 
      emtuple = NULL;
      Py_DECREF(emdict);
      emdict = NULL;
      if (!newList)
        return NULL;

      NAME_CAST_TO(_ListType, newList, cList)
      for(iterator si=aList->begin()+start, sei=aList->begin()+stop; si!=sei; si++)
        cList->push_back(*si);

      return newList;
    }
      
    catch (exception err)
      { Py_XDECREF(emtuple);
        Py_XDECREF(emdict);
        Py_XDECREF(newList);
        PYERROR(PyExc_Exception, err.what(), PYNULL);
      }
  }


/* -------------------------------------------------------------------------------------------------- */

  static PyObject *_reverse(TPyOrange *self)
  { PyTRY
      CAST_TO(_ListType, aList);
      std::reverse(aList->begin(), aList->end());
      RETURN_NONE;
    PyCATCH
  }


  static PyObject *_reduce(TPyOrange *self)
  { 
    PyTRY

      PyObject *res = Orange__reduce__((PyObject *)self, NULL, NULL);
      if (!res)
        return NULL;

      CAST_TO(_ListType, aList)
      if (aList->size()) {
        _PyTuple_Resize(&res, 4);
        PyTuple_SET_ITEM(res, 3, PySeqIter_New((PyObject *)self));
      }

      return res;
    PyCATCH
  }
};



template<class _WrappedListType, class _ListType, class _WrappedElement, TOrangeType *_PyElementType>
class ListOfWrappedMethods : public CommonListMethods<_WrappedListType, _ListType> {
public:
  typedef typename _ListType::iterator iterator;
  typedef typename _ListType::const_iterator const_iterator;

  static bool _fromPython(PyObject *obj, _WrappedElement &res)
  { if (obj == Py_None) {
      res = _WrappedElement();
      return true;
    }
  
    if (!obj || !PyObject_TypeCheck(obj, (PyTypeObject *)_PyElementType)) {
      if (_PyElementType->ot_inherited.tp_new) {
        PyObject *pyel = objectOnTheFly(obj, (PyTypeObject *)_PyElementType);
        if (pyel) {
          res = PyOrange_AS_Orange(pyel);
          return true;
        }
      }
        
      PyErr_Format(PyExc_TypeError, "expected '%s', got '%s'", _PyElementType->ot_inherited.tp_name, obj ? obj->ob_type->tp_name : "NULL");
      res = _WrappedElement();
      return false;
    }
  
    res = _WrappedElement(PyOrange_AS_Orange(obj));
    return true;
  }

  static _WrappedListType P_FromArguments(PyObject *arg, PyTypeObject *type = (PyTypeObject *)&PyOrOrange_Type)
  { PyObject *iterator = PyObject_GetIter(arg);
    if (!iterator) {
      PyErr_Format(PyExc_TypeError, "invalid arguments for '%s' constructor (sequence expected)", TYPENAME(typeid(_ListType)));
      return _WrappedListType();
    }

    _WrappedListType aList = _WrappedListType(mlnew _ListType(), type);

    int i = 0;
    for(PyObject *item = PyIter_Next(iterator); item; item = PyIter_Next(iterator), i++) {
      _WrappedElement obj;
      if (!_fromPython(item, obj)) {
        PyErr_Format(PyExc_TypeError, "element at index %i is of wrong type ('%s')", i, item->ob_type->tp_name);
        Py_DECREF(item);
        Py_DECREF(iterator);
        return _WrappedListType();
      }
      Py_DECREF(item);
      aList->push_back(obj);
    }

    return aList;
  }

  
  static PyObject *_FromArguments(PyTypeObject *type, PyObject *arg)
  { _WrappedListType obj = P_FromArguments(arg, type);
    if (!obj)
      return NULL;
    else
      return WrapOrange(obj);
  }


  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *)
  { if (!args || (PySequence_Check(args) && !PySequence_Size(args)))
      return _CreateEmptyList(type);

    if (PyTuple_Check(args) && PyTuple_Size(args)==1) {
      PyObject *arg=PyTuple_GetItem(args, 0);
      if (PySequence_Check(arg))
        return _FromArguments(type, arg);
    }

    return _FromArguments(type, args);
  }

  static PyObject *_getitem(TPyOrange *self, Py_ssize_t index)
  { PyTRY
      CAST_TO(_ListType, aList)
      return checkIndex(index, aList->size()) ? WrapOrange(aList->operator[](index)) : PYNULL;
    PyCATCH
  }

  static int _setitem(TPyOrange *self, Py_ssize_t index, PyObject *item)
  { PyTRY
      CAST_TO_err(_ListType, aList, -1)
      if (!checkIndex(index, aList->size()))
        return -1;
      if (item==NULL) {
        aList->erase(aList->begin()+index);
      }
      else {
        _WrappedElement citem;
        if (!_fromPython(item, citem))
          return -1;
        aList->operator[](index)=citem;
      }

      return 0;
    PyCATCH_1
  }

  static int _setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *args)
  { PyTRY
      CAST_TO_err(_ListType, aList, -1)
      if (!checkIndices(start, stop, aList->size()))
        return -1;

      if (args==NULL) {
        aList->erase(aList->begin()+start, aList->begin()+stop);
        return 0;
       }

      PyObject *emdict = NULL, *newList = NULL;
      try {
        PyObject *emdict = PyDict_New();
        PyObject *newList = _new(self->ob_type, args, emdict);
        Py_DECREF(emdict); emdict=NULL;
        if (!newList)
          return -1;

        NAME_CAST_TO_err(_ListType, newList, nList, -1)
        aList->erase(aList->begin()+start, aList->begin()+stop);
        aList->insert(aList->begin()+start, nList->begin(), nList->end());

        Py_DECREF(newList);
        newList = NULL;
        return 0;
      }

      catch (exception err)
        { Py_XDECREF(emdict);
          Py_XDECREF(newList);
          throw;
        }
    PyCATCH_1
  }

  static PyObject *_richcmp(TPyOrange *self, PyObject *other, int op)
  { PyTRY
      PyObject *myItem = NULL, *hisItem = NULL;
      try {
        if (!PySequence_Check(other)) {
          Py_INCREF(Py_NotImplemented);
          return Py_NotImplemented;
	      }

        CAST_TO(_ListType, aList)
        int myLen = aList->size();
        Py_ssize_t hisLen = PySequence_Size(other);

        if (myLen != hisLen) {
          if (op == Py_EQ) {
            Py_INCREF(Py_False);
            return Py_False;
          }
          if (op == Py_NE) {
            Py_INCREF(Py_True);
            return Py_True;
          }
        }

        Py_ssize_t len = myLen < hisLen ? myLen : hisLen;
        int k = 0;
        iterator ii(aList->begin());
        for (Py_ssize_t pos=0; !k && (pos<len); pos++) {
          myItem = WrapOrange(*(ii++));
          hisItem = PySequence_GetItem(other, pos);
          k = PyObject_RichCompareBool(myItem, hisItem, Py_NE);
          if (k<=0) {
            Py_DECREF(myItem);
            Py_DECREF(hisItem);
            myItem = NULL;
            hisItem = NULL;
          }
        }
        
        if (k == -1)
          return PYNULL;

        if (!k) {
          bool cmp;
          switch (op) {
            case Py_LT: cmp = myLen <  hisLen; break;
            case Py_LE: cmp = myLen <= hisLen; break;
            case Py_EQ: cmp = myLen == hisLen; break;
            case Py_NE: cmp = myLen != hisLen; break;
            case Py_GT: cmp = myLen >  hisLen; break;
            case Py_GE: cmp = myLen >= hisLen; break;
            default: return PYNULL; /* cannot happen */
          }
          PyObject *res = cmp ? Py_True : Py_False;
          Py_INCREF(res);
          return res;
        }

        // Here, myItem and hisItem are not decrefed yet!
        PyObject *res = PYNULL;
        if (op == Py_EQ)
          res = Py_False;
        else if (op == Py_NE)
          res = Py_True;
        else 
          res = PyObject_RichCompare(myItem, hisItem, op);

        Py_DECREF(myItem);
        Py_DECREF(hisItem);
        return res;
      }
      catch (exception err) {
        Py_XDECREF(myItem);
        Py_XDECREF(hisItem);
        throw;
      }
    PyCATCH
  }

  static PyObject *_str(TPyOrange *self)
  { 
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;
    
    CAST_TO(_ListType, aList);
    string res("<");
    for(iterator bi(aList->begin()), ei(bi), ee(aList->end()); ei!=ee; ei++) {
      if (ei!=bi)
        res += ", ";

      PyObject *obj = WrapOrange(*ei);
      PyObject *repred = PyObject_Str(obj);
      res += PyString_AsString(repred);
      Py_DECREF(obj);
      Py_DECREF(repred);
    }
    res += ">";
    return PyString_FromString(res.c_str());
  }

/* -------------------------------------------------------------------------------------------------- */

  static PyObject *_append(TPyOrange *self, PyObject *item)
  { PyTRY
      _WrappedElement obj;
      if (!_fromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      aList->push_back(obj);
      RETURN_NONE;
    PyCATCH
  }

  static PyObject *_count(TPyOrange *self, PyObject *item)
  { PyTRY
      _WrappedElement obj;
      if (!_fromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      int cnt=0;
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          cnt++;
      return PyInt_FromLong(cnt);
    PyCATCH
  }


  static int _contains(TPyOrange *self, PyObject *item)
  { PyTRY
      _WrappedElement obj;
      if (!_fromPython(item, obj))
        return -1;

      CAST_TO_err(_ListType, aList, -1);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          return 1;
      return 0;
    PyCATCH_1
  }


  static PyObject *_filter(TPyOrange *self, PyObject *args)
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

      CAST_TO(_ListType, aList)
      NAME_CAST_TO(_ListType, newList, cList)
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++) {
        PyObject *lel=WrapOrange(*bi);
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


  static PyObject *_index(TPyOrange *self, PyObject *item)
  { PyTRY
      _WrappedElement obj;
      if (!_fromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          return PyInt_FromLong(bi-aList->begin());
      PYERROR(PyExc_ValueError, "list.index(x): x not in list", PYNULL)
    PyCATCH
  }


  static PyObject *_insert(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList);

      PyObject *obj;
      int iindex;
      _WrappedElement item;

      if (!PyArg_ParseTuple(args, "iO", &iindex, &obj))
		  return NULL;

	  Py_ssize_t index = iindex;
      if (   !checkIndex(index, aList->size())
          || !_fromPython(obj, item))
        return PYNULL;
      
      aList->insert(aList->begin()+index, item);
      RETURN_NONE;
    PyCATCH
  }

  static PyObject *_native(TPyOrange *self)
  { PyTRY
      CAST_TO(_ListType, aList);
      PyObject *newList = PyList_New(aList->size());

      Py_ssize_t i=0;
      for(iterator li = aList->begin(), le = aList->end(); li!=le; li++)
        PyList_SetItem(newList, i++, WrapOrange(*li));

      return newList;
    PyCATCH
  }


  static PyObject *_pop(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList);
      int idx=aList->size()-1;
      if (!PyArg_ParseTuple(args, "|i:pop", &idx))
        return PYNULL;

      PyObject *ret=_getitem(self, idx);
      if (!ret)
        return PYNULL;

      aList->erase(aList->begin()+idx);
      return ret;
    PyCATCH
  }


  static PyObject *_remove(TPyOrange *self, PyObject *item)
  { PyTRY
      _WrappedElement obj;
      if (!_fromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi) {
          aList->erase(bi);
          RETURN_NONE;
        }
      PYERROR(PyExc_ValueError, "remove(x): x not in list", PYNULL)
    PyCATCH
  }


  class TCmpByCallback
  { public:
      PyObject *cmpfunc;

      TCmpByCallback(PyObject *func)
      { if (!PyCallable_Check(func))
          raiseErrorWho("CmpByCallback", "compare object not callable");

        cmpfunc=func;
        Py_INCREF(cmpfunc);
      }

      TCmpByCallback(const TCmpByCallback &other)
        : cmpfunc(other.cmpfunc)
      { Py_INCREF(cmpfunc); }

      ~TCmpByCallback()
      { Py_DECREF(cmpfunc); 
      }

      bool operator()(const _WrappedElement &x, const _WrappedElement &y) const
      { PyObject *pyx=WrapOrange(const_cast<_WrappedElement &>(x)), *pyy=WrapOrange(const_cast<_WrappedElement &>(y));
        PyObject *cmpres=PyObject_CallFunction(cmpfunc, "OO", pyx, pyy);
        Py_DECREF(pyx);
        Py_DECREF(pyy);

        if (!cmpres)
          throw pyexception();

        int res=PyInt_AsLong(cmpres);
        Py_DECREF(cmpres);

        return res<0;
      }
  };

  static PyObject *_concat(TPyOrange *self, PyObject *obj)
  { PyTRY
      CAST_TO(_ListType, aList);
      PyObject *newList = _new(self->ob_type, (PyObject *)self, NULL);
      if (!newList || (_setslice((TPyOrange *)newList, aList->size(), aList->size(), obj) == -1)) {
        Py_XDECREF(newList);
        return PYNULL;
      }
      else
        return newList;
    PyCATCH
  }

  static PyObject *_extend(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList)
      if (_setslice(self, aList->size(), aList->size(), args) == -1)
        return NULL;
      RETURN_NONE;
    PyCATCH
  }

  static PyObject *_sort(TPyOrange *self, PyObject *args)
  { 
    PyTRY
      PyObject *cmpfunc=NULL;
      if (!PyArg_ParseTuple(args, "|O:sort", &cmpfunc))
        return PYNULL;

      CAST_TO(_ListType, aList)
      if (cmpfunc)
        std::sort(aList->begin(), aList->end(), TCmpByCallback(cmpfunc));
      else
        std::sort(aList->begin(), aList->end());

      RETURN_NONE;
    PyCATCH;
  }
};


template<class _WrappedListType, class _ListType, class _Element>
class ListOfUnwrappedMethods : public CommonListMethods<_WrappedListType, _ListType> {
public:
  typedef typename _ListType::iterator iterator;
  typedef typename _ListType::const_iterator const_iterator;

  static _WrappedListType P_FromArguments(PyObject *arg)
  { if (!PySequence_Check(arg)) {
      PyErr_Format(PyExc_TypeError, "invalid arguments for '%s' constructor (sequence expected)", TYPENAME(typeid(_ListType)));
      return _WrappedListType();
    }

    _WrappedListType aList = mlnew _ListType();
    for(Py_ssize_t i=0, e=PySequence_Size(arg); i!=e; i++) {
      PyObject *pyobj=PySequence_GetItem(arg, i);
      _Element item;
      bool ok = convertFromPython(pyobj, item);
      if (!ok) {
        PyErr_Format(PyExc_TypeError, "element at index %i is of wrong type ('%s')", i, pyobj ? pyobj->ob_type->tp_name : "None");
        Py_DECREF(pyobj);
        return _WrappedListType();
      }
      Py_DECREF(pyobj);
      aList->push_back(item);
    }

    return aList;
  }

  
  static PyObject *_FromArguments(PyTypeObject *type, PyObject *arg)
  { _WrappedListType newList = P_FromArguments(arg);
    return newList ? WrapOrange(newList) : PYNULL;
  };


  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *)
  { if (!args || (PySequence_Check(args) && !PySequence_Size(args)))
      return _CreateEmptyList(type);

    if (PyTuple_Check(args) && PyTuple_Size(args)==1) {
      PyObject *arg=PyTuple_GetItem(args, 0);
      if (PySequence_Check(arg))
        return _FromArguments(type, arg);
    }

    return _FromArguments(type, args);
  }

  static PyObject *_getitem(TPyOrange *self, Py_ssize_t index)
  { PyTRY
      CAST_TO(_ListType, aList)
      return checkIndex(index, aList->size()) ? convertToPython(aList->operator[](index)) : PYNULL;
    PyCATCH
  }

  static int _setitem(TPyOrange *self, Py_ssize_t index, PyObject *item)
  { PyTRY
      CAST_TO_err(_ListType, aList, -1)
      if (!checkIndex(index, aList->size()))
        return -1;
      if (item==NULL) {
        aList->erase(aList->begin()+index);
      }
      else {
        _Element citem;
        if (!convertFromPython(item, citem))
          return -1;
        aList->operator[](index)=citem;
      }

      return 0;
    PyCATCH_1
  }

  static int _setslice(TPyOrange *self, Py_ssize_t start, Py_ssize_t stop, PyObject *args)
  { PyTRY
      CAST_TO_err(_ListType, aList, -1)
      if (!checkIndices(start, stop, aList->size()))
        return -1;

      if (args==NULL) {
        aList->erase(aList->begin()+start, aList->begin()+stop);
        return 0;
       }

      PyObject *emdict = NULL, *newList = NULL;
      try {
        PyObject *emdict = PyDict_New();
        PyObject *newList = _new(self->ob_type, args, emdict);
        Py_DECREF(emdict); emdict=NULL;
        if (!newList)
          return -1;

        NAME_CAST_TO_err(_ListType, newList, nList, -1)
        aList->erase(aList->begin()+start, aList->begin()+stop);
        aList->insert(aList->begin()+start, nList->begin(), nList->end());

        Py_DECREF(newList);
        newList = NULL;
        return 0;
      }

      catch (exception err)
        { Py_XDECREF(emdict);
          Py_XDECREF(newList);
          throw;
        }
    PyCATCH_1
  }


  static PyObject *_richcmp(TPyOrange *self, PyObject *other, int op)
  { PyTRY
      PyObject *myItem = NULL, *hisItem = NULL;
      try {
        if (!PySequence_Check(other)) {
          Py_INCREF(Py_NotImplemented);
          return Py_NotImplemented;
	      }

        CAST_TO(_ListType, aList)
        int myLen = aList->size();
        Py_ssize_t hisLen = PySequence_Size(other);

        if (myLen != hisLen) {
          if (op == Py_EQ) {
            Py_INCREF(Py_False);
            return Py_False;
          }
          if (op == Py_NE) {
            Py_INCREF(Py_True);
            return Py_True;
          }
        }

        Py_ssize_t len = myLen < hisLen ? myLen : hisLen;
        int k = 0;
        iterator ii(aList->begin());
        for (Py_ssize_t pos=0; !k && (pos<len); pos++) {
          myItem = convertToPython(*(ii++));
          hisItem = PySequence_GetItem(other, pos);
          k = PyObject_RichCompareBool(myItem, hisItem, Py_NE);
          if (k<=0) {
            Py_DECREF(myItem);
            Py_DECREF(hisItem);
            myItem = NULL;
            hisItem = NULL;
          }
        }
        
        if (k == -1)
          return PYNULL;

        if (!k) {
          bool cmp;
          switch (op) {
            case Py_LT: cmp = myLen <  hisLen; break;
            case Py_LE: cmp = myLen <= hisLen; break;
            case Py_EQ: cmp = myLen == hisLen; break;
            case Py_NE: cmp = myLen != hisLen; break;
            case Py_GT: cmp = myLen >  hisLen; break;
            case Py_GE: cmp = myLen >= hisLen; break;
            default: return PYNULL; /* cannot happen */
          }
          PyObject *res = cmp ? Py_True : Py_False;
          Py_INCREF(res);
          return res;
        }

        // Here, myItem and hisItem are not decrefed yet!
        PyObject *res = PYNULL;
        if (op == Py_EQ)
          res = Py_False;
        else if (op == Py_NE)
          res = Py_True;
        else 
          res = PyObject_RichCompare(myItem, hisItem, op);

        Py_DECREF(myItem);
        Py_DECREF(hisItem);
        return res;
      }
      catch (exception err) {
        Py_XDECREF(myItem);
        Py_XDECREF(hisItem);
        throw;
      }
    PyCATCH
  }


  static PyObject *_str(TPyOrange *self)
  { CAST_TO(_ListType, aList);
    string res("<");
    for(const_iterator bi(aList->begin()), ei(bi), ee(aList->end()); ei!=ee; ei++) {
      if (ei!=bi)
        res += ", ";

      PyObject *obj = convertToPython(*ei);
      PyObject *repred = PyObject_Str(obj);
      res += PyString_AsString(repred);
      Py_DECREF(obj);
      Py_DECREF(repred);
    }
    res += ">";
    return PyString_FromString(res.c_str());
  }

/* -------------------------------------------------------------------------------------------------- */

  static PyObject *_append(TPyOrange *self, PyObject *item)
  { PyTRY
      _Element obj;
      if (!convertFromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      aList->push_back(obj);
      RETURN_NONE;
    PyCATCH
  }

  static PyObject *_count(TPyOrange *self, PyObject *item)
  { PyTRY
      _Element obj;
      if (!convertFromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      int cnt=0;
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          cnt++;
      return PyInt_FromLong(cnt);
    PyCATCH
  }


  static int _contains(TPyOrange *self, PyObject *item)
  { PyTRY
      _Element obj;
      if (!convertFromPython(item, obj))
        return -1;

      CAST_TO_err(_ListType, aList, -1);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          return 1;
      return 0;
    PyCATCH_1
  }


  static PyObject *_filter(TPyOrange *self, PyObject *args)
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

      CAST_TO(_ListType, aList)
      NAME_CAST_TO(_ListType, newList, cList)
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++) {
        PyObject *lel=convertToPython(*bi);
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


  static PyObject *_index(TPyOrange *self, PyObject *item)
  { PyTRY
      _Element obj;
      if (!convertFromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi)
          return PyInt_FromLong(bi-aList->begin());
      PYERROR(PyExc_ValueError, "list.index(x): x not in list", PYNULL)
    PyCATCH
  }


  static PyObject *_insert(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList);

      PyObject *obj;
      int iindex;
      _Element item;

      if (!PyArg_ParseTuple(args, "iO", &iindex, &obj))
		  return NULL;

	  Py_ssize_t index = iindex;
	  if (   !checkIndex(index, aList->size())
          || !(convertFromPython(obj, item)))
        return PYNULL;
      
      aList->insert(aList->begin()+index, item);
      RETURN_NONE;
    PyCATCH
  }


  static PyObject *_native(TPyOrange *self)
  { PyTRY
      CAST_TO(_ListType, aList);
      PyObject *newList = PyList_New(aList->size());

      Py_ssize_t i=0;
      for(const_iterator li = aList->begin(), le = aList->end(); li!=le; li++)
        PyList_SetItem(newList, i++, convertToPython(*li));

      return newList;
    PyCATCH
  }


  static PyObject *_pop(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList);
      int idx=aList->size()-1;
      if (!PyArg_ParseTuple(args, "|i:pop", &idx))
        return PYNULL;

      PyObject *ret=_getitem(self, idx);
      if (!ret)
        return PYNULL;

      aList->erase(aList->begin()+idx);
      return ret;
    PyCATCH
  }


  static PyObject *_remove(TPyOrange *self, PyObject *item)
  { PyTRY
      _Element obj;
      if (!convertFromPython(item, obj))
        return PYNULL;

      CAST_TO(_ListType, aList);
      for(iterator bi=aList->begin(), be=aList->end(); bi!=be; bi++)
        if (obj==*bi) {
          aList->erase(bi);
          RETURN_NONE;
        }
      PYERROR(PyExc_ValueError, "remove(x): x not in list", PYNULL)
    PyCATCH
  }


  class TCmpByCallback
  { public:
      PyObject *cmpfunc;

      TCmpByCallback(PyObject *func)
      { if (!PyCallable_Check(func))
          raiseErrorWho("CmpByCallback", "compare object not callable");

        cmpfunc=func;
        Py_INCREF(cmpfunc);
      }

      TCmpByCallback(const TCmpByCallback &other)
        : cmpfunc(other.cmpfunc)
      { Py_INCREF(cmpfunc); }

      ~TCmpByCallback()
      { Py_DECREF(cmpfunc); 
      }

      bool operator()(const _Element &x, const _Element &y) const
      { PyObject *pyx=convertToPython(x), *pyy=convertToPython(y);
        PyObject *cmpres=PyObject_CallFunction(cmpfunc, "OO", pyx, pyy);
        Py_DECREF(pyx);
        Py_DECREF(pyy);

        if (!cmpres)
          throw pyexception();

        int res=PyInt_AsLong(cmpres);
        Py_DECREF(cmpres);

        return res<0;
      }
  };

  static PyObject *_concat(TPyOrange *self, PyObject *obj)
  { PyTRY
      CAST_TO(_ListType, aList);
      PyObject *newList = _new(self->ob_type, (PyObject *)self, NULL);
      if (!newList || (_setslice((TPyOrange *)newList, aList->size(), aList->size(), obj) == -1)) {
        Py_XDECREF(newList);
        return PYNULL;
      }
      else
        return newList;
    PyCATCH
  }


  static PyObject *_extend(TPyOrange *self, PyObject *args)
  { PyTRY
      CAST_TO(_ListType, aList)
      if (_setslice(self, aList->size(), aList->size(), args) == -1)
        return NULL;
      RETURN_NONE;
    PyCATCH
  }


  static PyObject *_sort(TPyOrange *self, PyObject *args)
  { 
    PyTRY
      PyObject *cmpfunc=NULL;
      if (!PyArg_ParseTuple(args, "|O:sort", &cmpfunc))
        return PYNULL;

      CAST_TO(_ListType, aList)
      if (cmpfunc)
        std::sort(aList->begin(), aList->end(), TCmpByCallback(cmpfunc));
      else
        std::sort(aList->begin(), aList->end());

      RETURN_NONE;
    PyCATCH;
  }
};

#endif
