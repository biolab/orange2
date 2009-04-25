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

#include "cls_orange.hpp"


PyObject *PyOrange_DictProxy_New(TPyOrange *bl)
{
  // Cannot implement print (crashes due to FILE *)
  // So I need to disable the inherited
  PyOrange_DictProxy_Type.tp_print = 0;

	PyObject *mp = PyDict_Type.tp_new(&PyOrange_DictProxy_Type, PYNULL, PYNULL);
  ((TPyOrange_DictProxy *)mp)->backlink = bl;
  bl->orange_dict = mp;

	return mp;
}



void PyOrange_DictProxy_dealloc(PyObject *mp)
{
  TPyOrange *backlink = ((TPyOrange_DictProxy *)mp)->backlink;
  if (backlink)
    backlink->orange_dict = NULL;
  PyDict_Type.tp_dealloc(mp);
}



PyObject *PyOrange_DictProxy_repr(TPyOrange_DictProxy *mp)
{
  string s = "{";
  bool notFirst = false;

  if (mp->backlink) {
    for (const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties; pd->name; pd++) {
      PyObject *pyval = Orange_getattr1(mp->backlink, pd->name);
      if (!pyval)
        return PYNULL;

      PyObject *pystr = PyObject_Repr(pyval);
      Py_DECREF(pyval);
      if (!pystr || !PyString_Check(pystr)) {
        Py_XDECREF(pystr);
        return PYNULL;
      }

      if (notFirst)
        s += ", ";
      else
        notFirst = true;

      s += "'";
      s += pd->name;
      s += "': ";
      s += PyString_AsString(pystr);

      Py_DECREF(pystr);
    }
  }

  PyObject *pystr = PyDict_Type.tp_repr((PyObject *)mp);
  if (!pystr || !PyString_Check(pystr)) {
    Py_XDECREF(pystr);
    return PYNULL;
  }

  if (notFirst && (PyString_Size(pystr) > 2))
    s += "; ";
  s += PyString_AsString(pystr) + 1;
  Py_DECREF(pystr);

  return PyString_FromString(s.c_str());
}


int PyOrange_DictProxy_length(TPyOrange_DictProxy *mp)
{
  int inlen = PyDict_Size((PyObject *)mp);
  if (mp->backlink) {
    const TPropertyDescription *ppd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties, *pd = ppd;
    while(pd->name)
      pd++;
    inlen += pd-ppd;
  }

  return inlen;
}


PyObject *PyOrange_DictProxy_subscript(TPyOrange_DictProxy *mp, PyObject *key)
{
  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

  if (mp->backlink) 
    return Orange_getattr(mp->backlink, key);
    
  return PyDict_Type.tp_as_mapping->mp_subscript((PyObject *)mp, key);
}


int PyOrange_DictProxy_ass_sub(TPyOrange_DictProxy *mp, PyObject *v, PyObject *w)
{
  if (!PyString_Check(v))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", -1);

  if (mp->backlink) 
    return Orange_setattrLow(mp->backlink, v, w, false);

  return PyDict_Type.tp_as_mapping->mp_ass_subscript((PyObject *)mp, v, w);
}


static PyObject *PyOrange_DictProxy_keys(TPyOrange_DictProxy *mp)
{
  PyObject *keys = PyDict_Keys((PyObject *)mp);

  if (mp->backlink)
    for(const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties; pd->name; pd++) {
      PyObject *pyname = PyString_FromString(pd->name);
      PyList_Append(keys, pyname);
      Py_DECREF(pyname);
    }

  return keys;
}


static PyObject *PyOrange_DictProxy_values(TPyOrange_DictProxy *mp)
{
  PyObject *values = PyDict_Values((PyObject *)mp);

  if (mp->backlink)
    for(const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties; pd->name; pd++) {
      PyObject *pyattr = Orange_getattr1(mp->backlink, pd->name);
      if (!pyattr) {
        Py_DECREF(values);
        return PYNULL;
      }
       
      PyList_Append(values, pyattr);
      Py_DECREF(pyattr);
    }

  return values;
}


static PyObject *PyOrange_DictProxy_items(TPyOrange_DictProxy *mp)
{
  PyObject *items = PyDict_Items((PyObject *)mp);

  if (mp->backlink)
    for(const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties; pd->name; pd++) {
      PyObject *pyattr = Orange_getattr1(mp->backlink, pd->name);
      if (!pyattr) {
        Py_DECREF(items);
        return PYNULL;
      }
      
      PyObject *bc = Py_BuildValue("sN", pd->name, pyattr);
      PyList_Append(items, bc);
      Py_DECREF(bc);
    }

  return items;
}



static PyObject *PyOrange_DictProxy_update(TPyOrange_DictProxy *mp, PyObject *seq2)
{
  PyObject *key, *value;
  Py_ssize_t pos = 0;

  while (PyDict_Next(seq2, &pos, &key, &value)) {
    if (!PyString_Check(key))
      PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

    int res = mp->backlink ? Orange_setattrLow(mp->backlink, key, value, false) : 1;
    if (   (res == -1)
        || ((res == 1) && (PyDict_SetItem((PyObject *)mp, key, value) == -1)))
      return PYNULL;
  }

  RETURN_NONE;
}


PyObject *PyOrange_DictProxy_has_key(TPyOrange_DictProxy *mp, PyObject *key)
{
  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

  if (mp->backlink) {
    char *name = PyString_AsString(key);
    const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties;
    while(pd->name && strcmp(pd->name, name))
      pd++;
    if (pd->name)
      return PyInt_FromLong(1);
  }

  return PyInt_FromLong(PyDict_GetItem((PyObject *)mp, key) ? 1 : 0);
}


PyObject *PyOrange_DictProxy_get(TPyOrange_DictProxy *mp, PyObject *args)
{
  PyObject *key;
  PyObject *failobj = Py_None;

  if (!PyArg_UnpackTuple(args, "get", 1, 2, &key, &failobj))
    return NULL;

  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

  if (mp->backlink) {
    PyObject *res = Orange_getattr(mp->backlink, key);
    if (res)
      return res;

    PyErr_Clear();
  }

  PyObject *val = PyDict_GetItem((PyObject *)mp, key);
  if (!val)
    val = failobj;

	Py_INCREF(val);
	return val;
}


PyObject *PyOrange_DictProxy_setdefault(TPyOrange_DictProxy *mp, PyObject *args)
{
  PyObject *key;
  PyObject *failobj = Py_None;

  if (!PyArg_UnpackTuple(args, "get", 1, 2, &key, &failobj))
    return NULL;

  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

  if (mp->backlink) {
    PyObject *val = Orange_getattr(mp->backlink, key);
    if (val)
      return val;

    PyErr_Clear();
  }

  PyObject *val = PyDict_GetItem((PyObject *)mp, key);
  if (!val) {
    val = failobj;
    PyDict_SetItem((PyObject *)mp, key, val);
  }

  Py_INCREF(val);
  return val;
}


PyObject *PyOrange_DictProxy_pop(TPyOrange_DictProxy *mp, PyObject *args)
{
  PyObject *key;
  PyObject *failobj = PYNULL;

  if (!PyArg_UnpackTuple(args, "get", 1, 2, &key, &failobj))
    return PYNULL;

  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", PYNULL);

  if (mp->backlink) {
    char *name = PyString_AsString(key);
    const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties;
    while(pd->name && strcmp(pd->name, name))
      pd++;
    if (pd->name)
      PYERROR(PyExc_KeyError, "cannot remove built-in attributes", PYNULL);
  }

  PyObject *val = PyDict_GetItem((PyObject *)mp, key);
  if (val) {
    Py_INCREF(val);
    PyDict_DelItem((PyObject *)mp, key);
    return val;
  }

  if (failobj) {
    Py_INCREF(failobj);
    return failobj;
  }

  PyErr_SetObject(PyExc_KeyError, key);
  return PYNULL;
}


static int PyOrange_DictProxy_contains(TPyOrange_DictProxy *mp, PyObject *key)
{
  if (!PyString_Check(key))
    PYERROR(PyExc_AttributeError, "object's attribute name must be string", -1);

  char *name = PyString_AsString(key);
  const TPropertyDescription *pd = PyOrange_AS_Orange(mp->backlink)->classDescription()->properties;
  while(pd->name && strcmp(pd->name, name))
    pd++;
  if (pd->name)
    return 1;

  return PyDict_GetItem((PyObject *)mp, key)!=NULL ? 1 : 0;
}


PyObject *PyOrange_DictProxyIter_new(TPyOrange_DictProxy *dict, binaryfunc select);

PyObject *select_key(PyObject *key, PyObject *value);
PyObject *select_value(PyObject *key, PyObject *value);
PyObject *select_item(PyObject *key, PyObject *value);

PyObject *PyOrange_DictProxy_iterkeys(TPyOrange_DictProxy *dict)
{	return PyOrange_DictProxyIter_new(dict, select_key); }

static PyObject *PyOrange_DictProxy_itervalues(TPyOrange_DictProxy *dict)
{ return PyOrange_DictProxyIter_new(dict, select_value); }

static PyObject *PyOrange_DictProxy_iteritems(TPyOrange_DictProxy *dict)
{	return PyOrange_DictProxyIter_new(dict, select_item); }

static PyObject *PyOrange_DictProxy_iter(TPyOrange_DictProxy *mp)
{ 	return PyOrange_DictProxyIter_new(mp, select_key); }


#define NO_METHOD(name) \
PyObject *PyOrange_DictProxy_##name(PyObject *, PyObject *, PyObject *) \
{ PYERROR(PyExc_AttributeError, "Orange dictionary proxy does not support "#name, PYNULL); }

NO_METHOD(popitem)
NO_METHOD(fromkeys)
NO_METHOD(copy)
NO_METHOD(clear)
#undef NO_METHOD

#define NO_METHOD(name) \
  {#name, (PyCFunction)PyOrange_DictProxy_##name, METH_VARARGS, ""},

static PyMethodDef PyOrange_DictProxy_methods[] = {
  {"has_key",    (PyCFunction)PyOrange_DictProxy_has_key,    METH_O, ""},
  {"get",        (PyCFunction)PyOrange_DictProxy_get,        METH_VARARGS, ""},
  {"setdefault", (PyCFunction)PyOrange_DictProxy_setdefault, METH_VARARGS, ""},
  {"pop",        (PyCFunction)PyOrange_DictProxy_pop,        METH_VARARGS, ""},
  {"keys",       (PyCFunction)PyOrange_DictProxy_keys,       METH_NOARGS, ""},
  {"items",      (PyCFunction)PyOrange_DictProxy_items,      METH_NOARGS, ""},
  {"values",     (PyCFunction)PyOrange_DictProxy_values,     METH_NOARGS, ""},
  {"update",     (PyCFunction)PyOrange_DictProxy_update,     METH_O, ""},
  {"iterkeys",   (PyCFunction)PyOrange_DictProxy_iterkeys,   METH_NOARGS, ""},
  {"itervalues", (PyCFunction)PyOrange_DictProxy_itervalues, METH_NOARGS, ""},
  {"iteritems",  (PyCFunction)PyOrange_DictProxy_iteritems,  METH_NOARGS, ""},
  NO_METHOD(popitem)
  NO_METHOD(fromkeys)
  NO_METHOD(copy)
  NO_METHOD(clear)
  #undef NO_METHOD
  {NULL,		NULL}	/* sentinel */
};


static PyMappingMethods PyOrange_DictProxy_as_mapping = {
	(inquiry)PyOrange_DictProxy_length,
	(binaryfunc)PyOrange_DictProxy_subscript, /*mp_subscript*/
	(objobjargproc)PyOrange_DictProxy_ass_sub, /*mp_ass_subscript*/
};


/* Hack to implement "key in dict" */
static PySequenceMethods PyOrange_DictProxy_as_sequence = {
	0,					/* sq_length */
	0,					/* sq_concat */
	0,					/* sq_repeat */
	0,					/* sq_item */
	0,					/* sq_slice */
	0,					/* sq_ass_item */
	0,					/* sq_ass_slice */
	(objobjproc)PyOrange_DictProxy_contains,		/* sq_contains */
	0,					/* sq_inplace_concat */
	0,					/* sq_inplace_repeat */
};


static PyObject *PyOrange_DictProxy_hash(PyObject *v, PyObject *w, int op)
{ PYERROR(PyExc_AttributeError, "Orange dictionary proxy is not hashable", PYNULL); }


PyTypeObject PyOrange_DictProxy_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,
	"Orange proxy dict",
	sizeof(TPyOrange_DictProxy),
	0,
	PyOrange_DictProxy_dealloc,  /* tp_dealloc */
	0,  /* tp_print */
	0,  /* tp_getattr */
	0,  /* tp_setattr */
	0,  /* tp_compare */
	(reprfunc)PyOrange_DictProxy_repr,
	0,
	&PyOrange_DictProxy_as_sequence,
	&PyOrange_DictProxy_as_mapping,
	0,  /* tp_hash */
	0,  /* tp_call */
	0,  /* tp_str */
	PyObject_GenericGetAttr,
	0,  /* tp_setattro */
	0,  /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |	Py_TPFLAGS_BASETYPE
	#ifdef Py_TPFLAGS_DICT_SUBCLASS
	| Py_TPFLAGS_DICT_SUBCLASS
	#endif
	,
	0,  /* tp_doc */
	(traverseproc)PyDict_Type.tp_traverse,
	(inquiry)PyDict_Type.tp_clear,
	0,  /* tp_richcompare */
	0,  /* tp_weaklistoffset */
	(getiterfunc)PyOrange_DictProxy_iter,
	0,  /* tp_iternext */
	PyOrange_DictProxy_methods,
	0,  /* tp_members */
	0,  /* tp_getset */
	&PyDict_Type,  /* tp_base */
	0,  /* tp_dict */
	0,  /* tp_descr_get */
	0,  /* tp_descr_set */
	0,  /* tp_dictoffset */
	0,  /* tp_init */
	PyType_GenericAlloc,
	0,  /* tp_new */
	_PyObject_GC_Del,
};



/* PyDictIter_Type has no Py_TPFLAGS_BASETYPE set, so I cannot
   derive the iterator. Most of the following code is base on
   the corresponding functions from dictobject.c. */

PyObject *select_key(PyObject *key, PyObject *value)
{
	Py_INCREF(key);
	return key;
}


PyObject *select_value(PyObject *key, PyObject *value)
{
	Py_INCREF(value);
	return value;
}


PyObject *select_item(PyObject *key, PyObject *value)
{
	PyObject *res = PyTuple_New(2);

	if (res != NULL) {
		Py_INCREF(key);
		Py_INCREF(value);
		PyTuple_SET_ITEM(res, 0, key);
		PyTuple_SET_ITEM(res, 1, value);
	}
	return res;
}


extern PyTypeObject PyOrange_DictProxyIter_Type;

class TPyOrange_DictyProxyIter {
public:
	PyObject_HEAD

	TPyOrange_DictProxy *di_dict; /* Set to NULL when iterator is exhausted */
	binaryfunc di_select;

  // This is for iteration over built-ins...
  const TPropertyDescription *pd;

  // ...and that's for the dictionary
	Py_ssize_t di_used;
	Py_ssize_t di_pos;
};


PyObject *PyOrange_DictProxyIter_new(TPyOrange_DictProxy *dict, binaryfunc select)
{
  TPyOrange_DictyProxyIter *di = PyObject_New(TPyOrange_DictyProxyIter, &PyOrange_DictProxyIter_Type);
  if (di == NULL)
    return NULL;

  Py_INCREF(dict);
  di->di_dict = dict;
  di->di_select = select;

  di->pd = dict->backlink ? PyOrange_AS_Orange(di->di_dict->backlink)->classDescription()->properties : NULL;

  di->di_used = dict->ma_used;
  di->di_pos = 0;

  return (PyObject *)di;
}


void PyOrange_DictProxyIter_dealloc(TPyOrange_DictyProxyIter *di)
{
  Py_XDECREF(di->di_dict);
  PyObject_Del(di);
}


static PyObject *PyOrange_DictProxyIter_iternext(TPyOrange_DictyProxyIter *di)
{
  PyObject *key, *value;

  if (di->di_dict == NULL)
    return NULL;


  if (di->pd) {
    if (!di->di_dict->backlink) {
      di->di_used = -1;
      PYERROR(PyExc_RuntimeError, "Orange object destroyed during iteration", PYNULL);
    }

    PyObject *res;

    if (di->di_select == select_key)
      res = PyString_FromString(di->pd->name);

    else {
      PyObject *value = Orange_getattr1(di->di_dict->backlink, di->pd->name);
      if (di->di_select == select_value)
        res = value;
      else {
        res = select_item(PyString_FromString(di->pd->name), value);
        Py_DECREF(value);
      }
    }

    if (!(++di->pd)->name)
      di->pd = NULL;

    return res;
  }

  else {
    if (di->di_used != di->di_dict->ma_used) {
      di->di_used = -1; /* Make this state sticky */
      PYERROR(PyExc_RuntimeError, "dictionary changed size during iteration", PYNULL);
    }

    if (PyDict_Next((PyObject *)(di->di_dict), &di->di_pos, &key, &value))
		  return (*di->di_select)(key, value);
  }

	Py_DECREF(di->di_dict);
	di->di_dict = NULL;
	return NULL;
}


// Could use Python's, but it only appeared in 2.3
// We can remove this when we stop caring about 2.2 users
PyObject *PyObject_MySelfIter(PyObject *obj)
{
	Py_INCREF(obj);
	return obj;
}

PyTypeObject PyOrange_DictProxyIter_Type = {
	PyObject_HEAD_INIT(&PyType_Type)
	0,					/* ob_size */
	"orange dictionary proxy-iterator",			/* tp_name */
	sizeof(TPyOrange_DictyProxyIter),			/* tp_basicsize */
	0,					/* tp_itemsize */
	/* methods */
	(destructor)PyOrange_DictProxyIter_dealloc, 		/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,			/* tp_flags */
 	0,					/* tp_doc */
 	0,					/* tp_traverse */
 	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	PyObject_MySelfIter,			/* tp_iter */
	(iternextfunc)PyOrange_DictProxyIter_iternext,	/* tp_iternext */
	0,					/* tp_methods */
	0,					/* tp_members */
	0,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
};
