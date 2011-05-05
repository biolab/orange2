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


#ifndef __MAPTEMPLATES_HPP
#define __MAPTEMPLATES_HPP

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#include "ormap.hpp"
#include "c2py.hpp"

template<class T>
static bool _orangeValueFromPython(PyObject *obj, T &val, PyTypeObject *PyKeyType)
{ if (!PyObject_TypeCheck(obj, PyKeyType)) {
    PyErr_Format(PyExc_TypeError, "invalid key: expected '%s', got '%s'", PyKeyType->tp_name, obj->ob_type->tp_name);
    return false;
  }
  val = PyOrange_AS_Orange(obj);
  return true;
}

template<class T>
static bool _nonOrangeValueFromPython(PyObject *obj, T &val, PyTypeObject *PyKeyType)
{ if (!convertFromPython(obj, val)) {
    PyErr_Format(PyExc_TypeError, "invalid key ('%s')", obj->ob_type->tp_name);
      return false;
  }
  
  return true;
}


template<class T>
static PyObject *_orangeValueToPython(const T &val)
{ return WrapOrange(const_cast<T &>(val)); }


template<class T>
static PyObject *_nonOrangeValueToPython(const T &val)
{ return convertToPython(const_cast<T &>(val)); }


#ifdef _MSC_VER
  #define INITIALIZE_MAPMETHODS(NAME, KEYTYPE, VALUETYPE, KFP, VFP, KTP, VTP) \
  TOrangeType *NAME::PyKeyType = KEYTYPE; \
  TOrangeType *NAME::PyValueType = VALUETYPE; \
  NAME::TKeyFromPython NAME::convertKeyFromPython = KFP; \
  NAME::TValueFromPython NAME::convertValueFromPython = VFP; \
  NAME::TKeyToPython NAME::convertKeyToPython = KTP; \
  NAME::TValueToPython NAME::convertValueToPython = VTP;
#else
  #define INITIALIZE_MAPMETHODS(NAME, KEYTYPE, VALUETYPE, KFP, VFP, KTP, VTP) \
  template <> TOrangeType *NAME::PyKeyType = KEYTYPE; \
  template <> TOrangeType *NAME::PyValueType = VALUETYPE; \
  template <> NAME::TKeyFromPython NAME::convertKeyFromPython = KFP; \
  template <> NAME::TValueFromPython NAME::convertValueFromPython = VFP; \
  template <> NAME::TKeyToPython NAME::convertKeyToPython = KTP; \
  template <> NAME::TValueToPython NAME::convertValueToPython = VTP;
#endif

template<class _WrappedMapType, class _MapType, class _Key, class _Value>
class MapMethods {
public:
  typedef pair<const _Key, _Value> _PairType;
  typedef map<_Key, _Value> mytype;
  typedef typename mytype::iterator iterator;
  typedef typename mytype::const_iterator const_iterator;

  typedef bool (*TKeyFromPython)(PyObject *, _Key &, PyTypeObject *);
  typedef bool (*TValueFromPython)(PyObject *, _Value &, PyTypeObject *);
  typedef PyObject *(*TKeyToPython)(const _Key &);
  typedef PyObject *(*TValueToPython)(const _Value &);

  static TOrangeType *PyKeyType;   // NULL for non-orange keys!
  static TOrangeType *PyValueType; // NULL for non-orange values!

  static TKeyFromPython convertKeyFromPython;
  static TValueFromPython convertValueFromPython;
  static TKeyToPython convertKeyToPython;
  static TValueToPython convertValueToPython;

  static bool _keyFromPython(PyObject *obj, _Key &key)
  { if (!obj) {
      PyErr_Format(PyExc_TypeError, "invalid key (NULL)");
      return false;
    }

    return convertKeyFromPython(obj, key, (PyTypeObject *)PyKeyType);
  }

  static bool _valueFromPython(PyObject *obj, _Value &value)
  { if (!obj) {
      PyErr_Format(PyExc_TypeError, "invalid value (NULL)");
      return false;
    }

    return convertValueFromPython(obj, value, (PyTypeObject *)PyValueType);
  }


  static bool updateLow(_MapType *uMap, PyObject *arg)
  {
    if (PyDict_Check(arg)) {
      Py_ssize_t pos=0;
      PyObject *pykey, *pyvalue;
      while (PyDict_Next(arg, &pos, &pykey, &pyvalue)) {
        if (_setitemlow(uMap, pykey, pyvalue)<0)
          return false;
      }
      return true;
    }


    PyObject *it = PyObject_GetIter(arg);
	  if (it == NULL)
      return _WrappedMapType();

	  for (int i = 0; ; ++i) {
		  PyObject *item = PyIter_Next(it);
		  if (item == NULL)
        if (!PyErr_Occurred()) {
          Py_DECREF(it);
          return true;
        }
			  else
          return false;

		  PyObject *fast = PySequence_Fast(item, "");
      Py_DECREF(item);

		  if (fast == NULL) {
			  if (PyErr_ExceptionMatches(PyExc_TypeError))
				  PyErr_Format(PyExc_TypeError,	"cannot convert dictionary update sequence element #%d to a sequence", i);
			  return false;
		  }

      const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
		  if (n != 2) {
        PyErr_Format(PyExc_ValueError, "dictionary update sequence element #%d has length %d; 2 is required", i, n);
        Py_DECREF(fast);
        return false;
		  }

      PyObject *pykey = PySequence_Fast_GET_ITEM(fast, 0);
      PyObject *pyvalue = PySequence_Fast_GET_ITEM(fast, 1);
      Py_DECREF(fast);

      if (_setitemlow(uMap, pykey, pyvalue)<0)
        return false;
	  }

    Py_DECREF(it);
    return true;
  }


  static _WrappedMapType P_FromArguments(PyObject *arg, PyTypeObject *type = (PyTypeObject *)&PyOrOrange_Type)
  { 
    _MapType *uMap = mlnew _MapType();
    _WrappedMapType aMap = _WrappedMapType(uMap, type);

    return updateLow(uMap, arg) ? aMap : _WrappedMapType();
  }

  
  static PyObject *_FromArguments(PyTypeObject *type, PyObject *arg)
  { _WrappedMapType obj = P_FromArguments(arg, type);
    return obj ? WrapOrange(obj) : NULL;
  }


  static PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *)
  { if (!args || (PySequence_Check(args) && !PySequence_Size(args)))
      return WrapNewOrange(mlnew _MapType(), type);

    if (PyTuple_Check(args) && PyTuple_Size(args)==1) {
      PyObject *arg = PyTuple_GetItem(args, 0);
      if (PySequence_Check(arg) || PyDict_Check(arg))
        return _FromArguments(type, arg);
    }

    return _FromArguments(type, args);
  }


  static bool findKey(_MapType *aMap, PyObject *pykey, iterator &fi, bool setError)
  {
    _Key key;
    if (!_keyFromPython(pykey, key))
      return false;

    fi = aMap->find(key);
    if (fi==aMap->end()) {
      if (setError) {
        PyObject *repred = PyObject_Str(pykey);
        PyErr_Format(PyExc_KeyError, PyString_AsString(repred));
        Py_DECREF(repred);
      }
      return false;
    }
    return true;
  }

  static PyObject *_getitem(TPyOrange *self, PyObject *pykey)
  { CAST_TO(_MapType, aMap)
    iterator fi;
    return findKey(aMap, pykey, fi, true) ? convertValueToPython((*fi).second) : PYNULL;
  }


  static int _setitemlow(_MapType *aMap, PyObject *pykey, PyObject *pyvalue)
  {
    _Key key;
    _Value value;
    if (!_keyFromPython(pykey, key) || !_valueFromPython(pyvalue, value))
      return -1;
  
      aMap->__ormap[key] = value;
    return 0;
  }


  static int _setitem(TPyOrange *self, PyObject *pykey, PyObject *pyvalue)
  { CAST_TO_err(_MapType, aMap, -1)

    if (pyvalue)
      return _setitemlow(aMap, pykey, pyvalue);
    
    iterator fi;
    if (!findKey(aMap, pykey, fi, true))
      return -1;

    aMap->erase(fi);
    return 0;
  }


  static PyObject *_str(TPyOrange *self)
  { 
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;

    CAST_TO(_MapType, aMap);
    string res("{");
    for(const_iterator bi(aMap->begin()), ei(bi), ee(aMap->end()); ei!=ee; ei++) {
      if (ei!=bi)
        res += ", ";

      PyObject *obj = convertKeyToPython((*ei).first);
      PyObject *repred = PyObject_Str(obj);
      res += PyString_AsString(repred);
      res += ": ";
      Py_DECREF(obj);
      Py_DECREF(repred);

      obj = convertValueToPython((*ei).second);
      repred = PyObject_Str(obj);
      res += PyString_AsString(repred);
      Py_DECREF(obj);
      Py_DECREF(repred);
    }
    res += "}";
    return PyString_FromString(res.c_str());
  }

  static Py_ssize_t _len(TPyOrange *self)
  { PyTRY
      CAST_TO_err(_MapType, aMap, -1);
      return aMap->size();
    PyCATCH_1
  }  


  static int _contains(TPyOrange *self, PyObject *pykey)
  { CAST_TO_err(_MapType, aMap, -1)
    iterator fi;
    return findKey(aMap, pykey, fi, false) ? 1 : 0;
  }

  
/* -------------------------------------------------------------------------------------------------- */

  static PyObject *_has_key(TPyOrange *self, PyObject *pykey)
  { const int cont = _contains(self, pykey);
    if (cont<0)
      return PYNULL;
    return PyInt_FromLong(cont);
  }


  static PyObject *_get(TPyOrange *self, PyObject *args)
  { PyObject *pykey;
    PyObject *deflt = Py_None;
    if (!PyArg_ParseTuple(args, "O|O:get", &pykey, &deflt))
		  return PYNULL;

    CAST_TO(_MapType, aMap)

    iterator fi;
    if (!findKey(aMap, pykey, fi, false)) {
      Py_INCREF(deflt);
      return deflt;
    }

    return convertValueToPython((*fi).second);
  }
      

  static PyObject *_setdefault(TPyOrange *self, PyObject *args)
  { PyObject *pykey;
    PyObject *deflt = Py_None;
    if (!PyArg_ParseTuple(args, "O|O:get", &pykey, &deflt))
		  return PYNULL;

    CAST_TO(_MapType, aMap)

    _Key key;
    if (!_keyFromPython(pykey, key))
      return PYNULL;

    iterator fi = aMap->find(key);
    if (fi==aMap->end()) {
      _Value value;
      if (!_valueFromPython(deflt, value))
        return PYNULL;
      aMap->__ormap[key] = value;
      Py_INCREF(deflt);
      return deflt;
    }

    return convertValueToPython((*fi).second);
  }


  static PyObject *_clear(TPyOrange *self)
  { CAST_TO(_MapType, aMap)
    aMap->clear();
    RETURN_NONE;
  }


  static PyObject *_keys(TPyOrange *self)
  { CAST_TO(_MapType, aMap)
    
    PyObject *res = PyList_New(aMap->size());
    Py_ssize_t i = 0;
    for(const_iterator ii(aMap->begin()), ie(aMap->end()); ii!=ie; ii++, i++) {
      PyObject *item = convertKeyToPython((*ii).first);
      if (!item) {
        Py_DECREF(res);
        return PYNULL;
      }

      PyList_SetItem(res, i, item);
    }

    return res;    
  }


  static PyObject *_values(TPyOrange *self)
  { CAST_TO(_MapType, aMap)
    
    PyObject *res = PyList_New(aMap->size());
    int i = 0;
    for(const_iterator ii(aMap->begin()), ie(aMap->end()); ii!=ie; ii++, i++) {
      PyObject *item = convertValueToPython((*ii).second);
      if (!item) {
        Py_DECREF(res);
        return PYNULL;
      }

      PyList_SetItem(res, i, item);
    }

    return res;    
  }


  static PyObject *_items(TPyOrange *self)
  { CAST_TO(_MapType, aMap)
    
    PyObject *res = PyList_New(aMap->size());
    Py_ssize_t i = 0;
    for(const_iterator ii(aMap->begin()), ie(aMap->end()); ii!=ie; ii++, i++) {
      PyObject *key = convertKeyToPython((*ii).first);
      PyObject *value = key ? convertValueToPython((*ii).second) : NULL;
      if (!value) {
        Py_DECREF(res);
        return PYNULL;
      }

      PyList_SetItem(res, i, Py_BuildValue("OO", key, value));
    }

    return res;    
  }

  static PyObject *_update(TPyOrange *self, PyObject *arg)
  { CAST_TO(_MapType, aMap)
    if (!updateLow(aMap, arg))
      return PYNULL;
    RETURN_NONE;    
  }


  static PyObject *_reduce(TPyOrange *self)
  { 
    PyTRY

      PyObject *res = Orange__reduce__((PyObject *)self, NULL, NULL);
      if (!res)
        return NULL;

      CAST_TO(_MapType, aMap)
      if (aMap->size()) {
        _PyTuple_Resize(&res, 5);

        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(res, 3, Py_None);

        PyObject *items = _items(self);
        PyTuple_SET_ITEM(res, 4, PySeqIter_New(items));
        Py_DECREF(items);
      }

      return res;
    PyCATCH
  }

};

#endif
