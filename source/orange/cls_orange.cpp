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


#include <string.h>

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"

#include "cls_value.hpp"
#include "cls_example.hpp"
#include "cls_orange.hpp"

#include "externs.px"


BASED_ON(Orange, ROOT)
DATASTRUCTURE(Orange, TPyOrange, orange_dict)
RECOGNIZED_ATTRIBUTES(Orange, "name shortDescription description")


PyObject *PyOrType_GenericAbstract(PyTypeObject *thistype, PyTypeObject *type, PyObject *args, PyObject *kwds)
{ PyTRY
    // if the user wants to create an instance of abstract class, we stop him
    if ((thistype == type) || !thistype->tp_base || !thistype->tp_base->tp_new) {
      PyErr_Format(PyExc_TypeError,  "cannot create instances of abstract class '%s'", type->tp_name);
      return NULL;
    }

    // if he derived a new class, we may let him
    return thistype->tp_base->tp_new(type, args, kwds);
  PyCATCH
}


PyObject *PyOrType_GenericNew(PyTypeObject *type, PyObject *args, PyObject *)
{ PyTRY
    PyObject *old = NULL;
    if (args && !PyArg_ParseTuple(args, "|O", &old)) {
      PyErr_Format(PyExc_TypeError, "%s: invalid arguments: nothing or an existing object expected", type->tp_name);
      return NULL;
    }

    if (old)
      if (PyType_IsSubtype(old->ob_type, type)) {
        Py_INCREF(old);
        return old;
      }
      else {
        PyErr_Format(PyExc_TypeError, "%s: '%s' is not a subtype of '%s'", type->tp_name, old->ob_type->tp_name, type->tp_name);
        return NULL;
      }

    // we assert PyOrange_OrangeBaseClass will succeed and that ot_defaultconstruct is defined.
    // the latter is pyxtract responsibility and pyxtract shouldn't be doubted  ;)
    POrange obj = PyOrange_OrangeBaseClass(type)->ot_defaultconstruct(type);
    if (!obj) {
      PyErr_Format(PyExc_SystemError, "constructor for '%s' failed", type->tp_name);
      return NULL;
    }
      
    return WrapOrange(obj);
  PyCATCH
}


PyObject *PyOrType_GenericNamedNew(PyTypeObject *type, PyObject *args, PyObject *)
{
  PyTRY
    PyObject *name=NULL;
    if (args && !PyArg_ParseTuple(args, "|O", &name)) {
      PyErr_Format(PyExc_TypeError, "%s: invalid arguments: nothing, a name or an existing object expected", type->tp_name);
      return NULL;
    }

    if (name && !PyString_Check(name))
      if (PyType_IsSubtype(name->ob_type, type)) {
        Py_INCREF(name);
        return name;
      }
      else {
        PyErr_Format(PyExc_TypeError, "%s: '%s' is not a subtype of '%s'", type->tp_name, name->ob_type->tp_name, type->tp_name);
        return NULL;
      }

    // we assert PyOrange_OrangeBaseClass will succeed and that ot_defaultconstruct is defined.
    // the latter is pyxtract responsibility and pyxtract shouldn't be doubted
    POrange obj = PyOrange_OrangeBaseClass(type)->ot_defaultconstruct(type);
    if (!obj) {
      PyErr_Format(PyExc_SystemError, "constructor for '%s' failed", type->tp_name);
      return NULL;
    }
      
    PyObject *self=WrapOrange(obj);

    if (!name || (PyObject_SetAttrString(self, "name", name)==0))
      return self;
    else {
      Py_DECREF(self);
      return PYNULL;
    }
  PyCATCH
}


int Orange_init(PyObject *self, PyObject *args, PyObject *keywords);


// Rewrapping: this is not a toy - use it cautiously
void rewrap(TPyOrange *&obj, PyTypeObject *type)
{ if (obj && (type!=obj->ob_type)) {
    if (obj->ob_refcnt>1) {
      #ifdef _MSC_VER
        throw exception("cannot rewrap (refcnt>1)");
      #else
        throw exception();
      #endif
    }
    PyObject_GC_UnTrack((PyObject *)obj);

    TPyOrange *newobj = (TPyOrange *)type->tp_alloc(type, 0);
    newobj->orange_dict = obj->orange_dict;
    newobj->ptr = obj->ptr;
    newobj->call_constructed = obj->call_constructed;
    newobj->is_reference = obj->is_reference;

    obj->orange_dict = NULL;
    obj->ptr = NULL;
    obj->freeRef();

    obj = newobj;
  }
}


PyObject *PyOrType_GenericCallableNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{ PyObject *self1 = NULL,
           *self2 = NULL,
           *ccReturn = NULL;

  PyTRY
  // we assert PyOrange_OrangeBaseClass will succeed and that ot_defaultconstruct is defined.
  // the latter is pyxtract responsibility and pyxtract shouldn't be doubted
    POrange obj=PyOrange_OrangeBaseClass(type)->ot_defaultconstruct(PyTuple_Size(args) ? (PyTypeObject *)&PyOrOrange_Type : type);
    if (!obj) {
      PyErr_Format(PyExc_SystemError, "constructor for '%s' failed", type->tp_name);
      goto err;
    }

    PyObject *self1 = WrapOrange(obj);

    if (!PyTuple_Size(args))
      return self1;

    else {
/*      if (self1->ob_type != type) {
        PyErr_Format(PyExc_SystemError, "Subclassed orange classes are not call-constructable", type->tp_name);
        Py_DECREF(self1);
        return NULL;
      }
*/
      if (!self1->ob_type->tp_call) {
        PyErr_Format(PyExc_SystemError, "error in orange class structure ('%s' not callable)", type->tp_name);
        goto err;
      }

      /* We should manually call init - Python will call init on self2, and it won't do anything.
         It's important to call it prior to setting call_constructed (below) */
      if (Orange_init(self1, args, kwds) < 0)
        goto err;

      /* this is to tell tp_call(self1) not to complain about keyword arguments;
         self1 is disposed later in this function, so this should cause no problems */
      ((TPyOrange *)self1)->call_constructed = true;
      PyObject *self2=self1->ob_type->tp_call(self1, args, kwds);

      if (self2) {
        if (type!=self1->ob_type) {
          char *oname = new char[30 + strlen(type->tp_name)];
          sprintf(oname, "_%s__call_construction_type", type->tp_name);
          PyObject *ccReturn = PyObject_GetAttrString((PyObject *)type, oname);
          delete oname;

          if (!ccReturn) {
            PyErr_Format(PyExc_SystemError, "no return type specified for call-construction of '%s'", type->tp_name);
            goto err;
          }

          if (!PyType_Check(ccReturn)) {
            PyErr_Format(PyExc_SystemError, "no return type specified for call-construction of '%s'", type->tp_name);
            goto err;
          }

          if (self2->ob_refcnt>1) { // Object might have a reference to itself...
            PyErr_Format(PyExc_SystemError, "cannot rewrap the class '%s' - too many references", self2->ob_type->tp_name);
            goto err;
          }

          rewrap((TPyOrange *&)self2, (PyTypeObject *)ccReturn);
        }

        Py_DECREF(self1);

        if (PyOrOrange_Check(self2))
          ((TPyOrange *)self2)->call_constructed = true;
        return self2;
      }
    
    }
  PyCATCH

err:
  Py_XDECREF(self1);
  Py_XDECREF(self2);
  Py_XDECREF(ccReturn);
  return NULL;
}


int Orange_traverse(TPyOrange *self, visitproc visit, void *arg)
{
  if (self->orange_dict) {
    int err = visit(self->orange_dict, arg);
    if (err)
      return err;
  }

  return ((TOrange *)self->ptr)->traverse(visit, arg);
}


int Orange_clear(TPyOrange *self)
{ 
  return ((TOrange *)self->ptr)->dropReferences();
}


PyObject *PyOrange__dict__(TPyOrange *self)
{ 
  if (!self->orange_dict)
    self->orange_dict = PyOrange_DictProxy_New(self);

  Py_INCREF(self->orange_dict);
  return (PyObject *)(self->orange_dict);
}


PyObject *PyOrange__members__(TPyOrange *self)
{
  const TPropertyDescription *ppd = PyOrange_AS_Orange(self)->classDescription()->properties;
  const TPropertyDescription *pd;
  for(pd = ppd; pd->name; pd++);

  PyObject *res = PyList_New(pd-ppd);
  for(pd = ppd; pd->name; pd++)
    PyList_SetItem(res, pd-ppd, PyString_FromString(pd->name));

  return res;
}


char *camel2underscore(const char *camel)
{
    const char *ci = camel;
    if ((*ci >= 'A') && (*ci <= 'Z')) {
        return NULL;
    }

    char *underscored = (char *)malloc(2*strlen(camel)+1);
    char *ui = underscored;
    bool changed = false;
    *ui = *ci;
    while(*ci) { // just copied
        if (   (*ci >= 'a') && (*ci <= 'z')       // a small letter
            && (ci[1] >= 'A') && (ci[1] <= 'Z')   // followed by capital
            && ((ci[2] < 'A') || (ci[2] > 'Z'))) { // not followed by capital 
            *++ui = '_';
            *++ui = *++ci + 32;
            changed = true;
        }
        else {
            *++ui = *++ci;
        }
    }
    if (!changed) {
        free(underscored);
        underscored = NULL;
    }
    return underscored;
}


PyObject *PyOrange_translateObsolete(PyObject *self, PyObject *pyname)
{ 
  char *name = PyString_AsString(pyname);
  char *underscored = camel2underscore(name);
  for(TOrangeType *selftype = PyOrange_OrangeBaseClass(self->ob_type); PyOrange_CheckType((PyTypeObject *)selftype); selftype=(TOrangeType *)(selftype->ot_inherited.tp_base)) {
      if (selftype->ot_aliases) {
          for(TAttributeAlias *aliases=selftype->ot_aliases; aliases->alias; aliases++) {
              if (!strcmp(name, aliases->alias) || (underscored && !strcmp(underscored, aliases->alias))) {
                  if (underscored) {
                      free(underscored);
                  }
                  return PyString_FromString(aliases->realName);
              }
          }
      }
  }
  if (underscored) {
      free(underscored);
  }
  return NULL;
}    


PyObject *Orange_getattr1(TPyOrange *self, const char *name)
// This is a getattr, without translation of obsolete names and without looking into the associated dictionary
{ PyTRY
    if (!self)
      PYERROR(PyExc_SystemError, "NULL Orange object", PYNULL);

    TOrange *me = (TOrange *)self->ptr;
    if (me->hasProperty(name)) {
      try {
        const TPropertyDescription *propertyDescription = me->propertyDescription(name);
        const type_info &propertyType = *propertyDescription->type;
        TPropertyTransformer *transformer = propertyDescription->transformer;

        if (propertyType==typeid(bool)) {
          bool value;
          me->getProperty(name, value);
          return transformer ? (PyObject *)transformer(&value) : PyBool_FromLong(value ? 1 : 0);
        }

        if (propertyType==typeid(int)) {
          int value;
          me->getProperty(name, value);
          return transformer ? (PyObject *)transformer(&value) : PyInt_FromLong(value);
        }

        if (propertyType==typeid(float)) {
          float value;
          me->getProperty(name, value);
          return transformer ? (PyObject *)transformer(&value) : PyFloat_FromDouble(value);
        }

        if (propertyType==typeid(string)) {
          string value;
          me->getProperty(name, value);
          return transformer ? (PyObject *)transformer(&value) : PyString_FromString(value.c_str());
        }

        if (propertyType==typeid(TValue)) {
          TValue value;
          me->getProperty(name, value);
          return transformer ? (PyObject *)transformer(&value) : Value_FromValue(value);
        }

        if (propertyType==typeid(TExample)) {
          POrange mlobj;
          me->wr_getProperty(name, mlobj);
          if (transformer)
            return (PyObject *)transformer(&mlobj);
          if (mlobj)
            return Example_FromWrappedExample(PExample(mlobj));
          RETURN_NONE;
        }
      
        POrange mlobj;
        me->wr_getProperty(name, mlobj);
        return transformer ? (PyObject *)transformer(&mlobj) : (PyObject *)WrapOrange(mlobj);
      } catch (exception err)
      {}
    }
 
    if (!strcmp(name, "name") || !strcmp(name, "shortDescription") || !strcmp(name, "description"))
      return PyString_FromString("");

    PyErr_Format(PyExc_AttributeError, "'%s' has no attribute '%s'", self->ob_type->tp_name, name);
    return PYNULL;
  PyCATCH;
}




PyObject *Orange_getattr1(TPyOrange *self, PyObject *pyname)
// This is a complete getattr, but without translation of obsolete names.
{ PyTRY
    if (!self)
      PYERROR(PyExc_SystemError, "NULL Orange object", PYNULL);

    if (self->orange_dict) {
      PyObject *res = PyDict_GetItem(self->orange_dict, pyname);
      if (res) {
        Py_INCREF(res);
        return res;
      }
    }
      
    PyObject *res = PyObject_GenericGetAttr((PyObject *)self, pyname);
    if (res)
      return res;

    PyErr_Clear();

    if (!PyString_Check(pyname))
      PYERROR(PyExc_TypeError, "object's attribute name must be a string", PYNULL);
    char *name=PyString_AsString(pyname);

    if (strcmp(name, "__dict__") == 0)
      return PyOrange__dict__(self);

    if (strcmp(name, "__members__") == 0)
      return PyOrange__members__(self);

    if (strcmp(name, "__class__") == 0) {
      Py_INCREF(self->ob_type);
      return (PyObject *)self->ob_type;
    }

    return Orange_getattr1(self, name);
  PyCATCH;
}


inline void PyDict_SIS_Steal(PyObject *dict, const char *name, PyObject *obj) {
  PyDict_SetItemString(dict, name, obj);
  Py_DECREF(obj);
}

PyObject *packOrangeDictionary(PyObject *self)
{
  PyTRY
    PyObject *packed = ((TPyOrange *)self)->orange_dict ? PyDict_Copy(((TPyOrange *)self)->orange_dict) : PyDict_New();

    TOrange *me = (TOrange *)((TPyOrange *)self)->ptr;

    for (const TPropertyDescription *pd = me->classDescription()->properties; pd->name; pd++) {
      if (!pd->readOnly) {
 
  //      const type_info &propertyType = pd->type;

        if (pd->type == &typeid(bool))
          PyDict_SIS_Steal(packed, pd->name, PyInt_FromLong(me->getProperty_bool(pd) ? 1 : 0));

        else if (pd->type == &typeid(int))
          PyDict_SIS_Steal(packed, pd->name, PyInt_FromLong(me->getProperty_int(pd)));

        else if (pd->type == &typeid(float))
          PyDict_SIS_Steal(packed, pd->name, PyFloat_FromDouble(me->getProperty_float(pd)));

        else if (pd->type == &typeid(string)) {
          string value;
          me->getProperty_string(pd, value);
          PyDict_SIS_Steal(packed, pd->name, PyString_FromString(value.c_str()));
        }

        else if (pd->type == &typeid(TValue)) {
          TValue value;
          me->getProperty_TValue(pd, value);
          PyDict_SIS_Steal(packed, pd->name, Value_FromValue(value));
        }

        else if (pd->type == &typeid(TExample)) {
          POrange mlobj;
          me->getProperty_POrange(pd, mlobj);
          if (mlobj)
            PyDict_SIS_Steal(packed, pd->name, Example_FromWrappedExample(PExample(mlobj)));
          else
            PyDict_SetItemString(packed, pd->name, Py_None);
        }
    
        else {
          POrange mlobj;
          me->getProperty_POrange(pd, mlobj);
          PyDict_SIS_Steal(packed, pd->name, (PyObject *)WrapOrange(mlobj));
        }
      }
    }

    return packed;
  PyCATCH
}


int Orange_setattr(TPyOrange *self, PyObject *pyname, PyObject *args);

int unpackOrangeDictionary(PyObject *self, PyObject *dict)
{
  PyObject *d_key, *d_value;
  Py_ssize_t i = 0;
  while (PyDict_Next(dict, &i, &d_key, &d_value)) {
//	  if (Orange_setattr1((TPyOrange *)self, d_key, d_value) == -1)
	  if (Orange_setattrLow((TPyOrange *)self, d_key, d_value, false) == -1)
	    return -1;
	}
  return 0;
}

ORANGE_API PyObject *Orange__reduce__(PyObject *self, PyObject *, PyObject *)
{
    if (!((TOrangeType *)(self->ob_type))->ot_constructorAllowsEmptyArgs) {
      PyErr_Format(PyExc_TypeError, "instances of type '%s' cannot be pickled", self->ob_type->tp_name);
      return NULL;
    }

    return Py_BuildValue("O()N", self->ob_type, packOrangeDictionary(self));
}



PyObject *objectOnTheFly(PyObject *args, PyTypeObject *objectType)
{
  PyObject *emptyDict = PyDict_New();
  PyObject *targs;
  if (PyTuple_Check(args)) {
    targs = args;
    Py_INCREF(targs);
  }
  else
    targs = Py_BuildValue("(O)", args);

  PyObject *obj = NULL;
  try {
    obj = objectType->tp_new(objectType, targs, emptyDict);
  }
  catch (...) {
    // do nothing; if it failed, the user probably didn't mean it
  }

  // If this failed, maybe the constructor actually expected a tuple...
  if (!obj && PyTuple_Check(args)) {
     PyErr_Clear();
     Py_DECREF(targs);
     targs = Py_BuildValue("(O)", args);
     try {
       obj = objectType->tp_new(objectType, targs, emptyDict);
     }
     catch (...) 
     {}
  }

  if (obj) {
    if (   objectType->tp_init != NULL
        && objectType->tp_init(obj, targs, emptyDict) < 0) {
          Py_DECREF(obj);
          obj = NULL;
    }
  }

  Py_DECREF(emptyDict);
  Py_DECREF(targs);

  return obj;
}


int Orange_setattr1(TPyOrange *self, char *name, PyObject *args)
{
  TOrange *me = (TOrange *)self->ptr;

  const TPropertyDescription *propertyDescription = me->propertyDescription(name, true);
  if (!propertyDescription)
    return 1;

  PyTRY
    if (propertyDescription->readOnly) {
      /* Property might be marked as readOnly, but have a specialized set function.
         The following code is pasted from PyObject_GenericSetAttr.
         If I'd call it here and the attribute is really read-only, PyObject_GenericSetAttr
         would blatantly store it in the dictionary. */
      PyObject *pyname = PyString_FromString(name);
      PyObject *descr = _PyType_Lookup(self->ob_type, pyname);
	    PyObject *f = PYNULL;
	    if (descr != NULL && PyType_HasFeature(descr->ob_type, Py_TPFLAGS_HAVE_CLASS)) {
		    descrsetfunc f = descr->ob_type->tp_descr_set;
		    if (f != NULL && PyDescr_IsData(descr))
			    return f(descr, (PyObject *)self, args);
      }

      PyErr_Format(PyExc_TypeError, "%s.%s: read-only attribute", self->ob_type->tp_name, name);
      return -1;
    }
  
    try {
      const type_info &propertyType = *propertyDescription->type;

      if ((propertyType==typeid(bool)) || (propertyType==typeid(int))) {
        int value;
        if (!PyArg_Parse(args, "i", &value)) {
          PyErr_Format(PyExc_TypeError, "invalid parameter type for %s.%s', (int expected)", self->ob_type->tp_name, name);
          return -1;
        }
        if (propertyType==typeid(bool))
          me->setProperty(name, value!=0);
        else
          me->setProperty(name, value);
        return 0;
      }

      if (propertyType==typeid(float)) {
        float value;
        if (!PyArg_Parse(args, "f", &value)) {
          PyErr_Format(PyExc_TypeError, "invalid parameter type for %s.%s', (float expected)", self->ob_type->tp_name, name);
          return -1;
        }
        me->setProperty(name, value);
        return 0;
      }

      if (propertyType==typeid(string)) {
        char *value;
        if (!PyArg_Parse(args, "s", &value)) {
          PyErr_Format(PyExc_TypeError, "invalid parameter type for %s.%s', (string expected)", self->ob_type->tp_name, name);
          return -1;
        }
        me->setProperty(name, string(value));
        return 0;
      }

      if (propertyType==typeid(TValue)) {
        TValue value;
        if (!convertFromPython(args, value))
          return -1;
        me->setProperty(name, value);
        return 0;
      }

      if (propertyType==typeid(TExample)) {
        if (args==Py_None) {
          me->wr_setProperty(name, POrange());
          return 0;
        }
        else {
          if (!PyOrExample_Check(args)) {
            PyErr_Format(PyExc_TypeError, "invalid parameter type for '%s.%s', (expected 'Example', got '%s')", self->ob_type->tp_name, name, args->ob_type->tp_name);
            return -1;
          }
          me->wr_setProperty(name, POrange(PyExample_AS_Example(args)));
          return 0;
        }
      }

      if (1/*propertyType==typeid(POrange)*/) {
        const type_info *wrappedType = propertyDescription->classDescription->type;

        PyTypeObject *propertyPyType=(PyTypeObject *)FindOrangeType(*wrappedType);
        if (!propertyPyType) {
          PyErr_Format(PyExc_SystemError, "Orange class %s, needed for '%s.%s' not exported to Python", TYPENAME(*wrappedType), self->ob_type->tp_name, name);
          return -1;
        }

        if (args==Py_None) {
          me->wr_setProperty(name, POrange());
          return 0;
        }

        // User might have supplied the correct object
        if (PyObject_TypeCheck(args, propertyPyType)) {
          me->wr_setProperty(name, PyOrange_AS_Orange((TPyOrange *)args));
          return 0;
        }

        // User might have supplied parameters from which we can construct the object
        if (propertyPyType->tp_new) {
          PyObject *obj = objectOnTheFly(args, propertyPyType);
          if (obj) {
            bool success = true;
            try {
              me->wr_setProperty(name, PyOrange_AS_Orange((TPyOrange *)obj));
            }
            catch (...) {
              success = false;
            }
            Py_DECREF(obj);
            if (success)
              return 0;
          }
        }

        PyErr_Format(PyExc_TypeError, "invalid parameter type for '%s.%s', (expected '%s', got '%s')", self->ob_type->tp_name, name, propertyPyType->tp_name, args->ob_type->tp_name);
        return -1;
      }

      PyErr_Format(PyExc_TypeError, "internal Orange error: unrecognized type '%s.%s'", self->ob_type->tp_name, name);
      return -1;
    } catch (exception err)
    {
      PyErr_Format(PyExc_TypeError, "error setting '%s.%s'", self->ob_type->tp_name, name);
      return -1;
    }
  PyCATCH_1
}


int Orange_setattr1(TPyOrange *self, PyObject *pyname, PyObject *args)
// This is a complete setattr, but without translation of obsolete names.
{ 
  if (!self)
    PYERROR(PyExc_SystemError, "NULL Orange object", -1);

  /* We first have to check for a specific handler.
     The following code is pasted from PyObject_GenericSetAttr, but we can't
     call it since it would store *all* attributes in the dictionary. */
  PyObject *descr = _PyType_Lookup(self->ob_type, pyname);
  PyObject *f = PYNULL;
  if (descr != NULL && PyType_HasFeature(descr->ob_type, Py_TPFLAGS_HAVE_CLASS)) {
    descrsetfunc f = descr->ob_type->tp_descr_set;
    if (f != NULL && PyDescr_IsData(descr))
      return f(descr, (PyObject *)self, args);
  }
 
  char *name=PyString_AsString(pyname);
  int res = Orange_setattr1(self, name, args);
  if (res != 1)
    return res;

  return 1; // attribute not set (not even attempted to), try something else
}


PyObject *Orange_new(PyTypeObject *type, PyObject *args, PyObject *keywords)  BASED_ON(ROOT, "()")
{ return WrapNewOrange(mlnew TOrange(), type); }

int Orange_init(PyObject *self, PyObject *, PyObject *keywords)
{ PyTRY
    return ((TPyOrange *)self)->call_constructed || SetAttr_FromDict(self, keywords, true) ? 0 : -1;
  PyCATCH_1
}


void Orange_dealloc(TPyOrange *self)
{
  if (!self->is_reference) {
    PyObject_GC_UnTrack((PyObject *)self);
    mldelete self->ptr;
  }

  // This may cause troubles in multithread use
  if (self->orange_dict) {
    ((TPyOrange_DictProxy *)self->orange_dict)->backlink = NULL;
    Py_DECREF(self->orange_dict);
  }

  self->ob_type->tp_free((PyObject *)self);
}



PyObject *Orange_getattr(TPyOrange *self, PyObject *name)
// This calls getattr1; first with the given, than with the translated name
{ 
  PyTRY
    PyObject *res = Orange_getattr1(self, name);
    char *underscored = NULL;
    if (!res) {
        char *camel = PyString_AsString(name);
        underscored = camel2underscore(camel);
        if (underscored) {
            PyObject *translation = PyString_FromString(underscored);
            PyErr_Clear();
            res = Orange_getattr1(self, translation);
            Py_DECREF(translation);
        }
    }
    if (!res) {
        PyObject *translation = PyOrange_translateObsolete((PyObject *)self, name);
        if (translation) {
            PyErr_Clear();
            res = Orange_getattr1(self, translation);
            Py_DECREF(translation);
        }
    }

    if (!res && underscored) {
        PyMethodDef *mi = self->ob_type->tp_methods;
        if (mi) {
            for(; mi->ml_name; mi++) {
                if (!strcmp(underscored, mi->ml_name)) {
                    res = PyMethod_New((PyObject *)mi->ml_meth, (PyObject *)self, (PyObject *)(self->ob_type));
                    break;
                }
            }
        }
    }

    if (underscored) {
        free(underscored);
    }
    return res;
  PyCATCH
}


int Orange_setattrDictionary(TPyOrange *self, const char *name, PyObject *args, bool warn)
{
  PyObject *pyname = PyString_FromString(name);
  int res = Orange_setattrDictionary(self, pyname, args, warn);
  Py_DECREF(pyname);
  return res;
}

int Orange_setattrDictionary(TPyOrange *self, PyObject* pyname, PyObject *args, bool warn)
{ PyTRY
    char *name = PyString_AsString(pyname);
    if (args) {
      /* Issue a warning unless name the name is in 'recognized_list' in some of the ancestors
         or the instance's class only derived from some Orange's class, but is written in Python */
      if (warn && PyOrange_CheckType(self->ob_type)) {
        char **recognized = NULL;
        for(PyTypeObject *otype = self->ob_type; otype && (!recognized || !*recognized); otype = otype->tp_base) {
          recognized = PyOrange_CheckType(otype) ? ((TOrangeType *)otype)->ot_recognizedattributes : NULL;
          if (recognized)
            for(; *recognized && strcmp(*recognized, name); recognized++);
        }

        if (!recognized || !*recognized) {
          char sbuf[255];
          sprintf(sbuf, "'%s' is not a builtin attribute of '%s'", name, self->ob_type->tp_name);
          if (PyErr_Warn(PyExc_OrangeAttributeWarning, sbuf))
            return -1;
        }
      }

      if (!self->orange_dict)
        self->orange_dict = PyOrange_DictProxy_New(self);

      return PyDict_SetItem(self->orange_dict, pyname, args);
    }
    else {
      if (self->orange_dict)
        return PyDict_DelItem(self->orange_dict, pyname);
      else {
        PyErr_Format(PyExc_AttributeError, "instance of '%s' has no attribute '%s'", self->ob_type->tp_name, name);
        return -1;
      }
    }
  PyCATCH_1
}

int Orange_setattrLow(TPyOrange *self, PyObject *pyname, PyObject *args, bool warn)
// This calls setattr1; first with the given, than with the translated name
{ PyTRY
    if (!PyString_Check(pyname))
      PYERROR(PyExc_AttributeError, "object's attribute name must be string", -1);

    // Try to set it as C++ class member
    int res = Orange_setattr1(self, pyname, args);
    if (res!=1)
      return res;
    
    PyErr_Clear();
    char *camel = PyString_AsString(pyname);
    char *underscored = camel2underscore(camel);
    if (underscored) {
        PyObject *translation = PyString_FromString(underscored);
        free(underscored);
        res = Orange_setattr1(self, translation, args);
        Py_DECREF(translation);
    }
    if (res!=1)
      return res;

    PyErr_Clear();
    // Try to translate it as an obsolete alias for C++ class member
    PyObject *translation = PyOrange_translateObsolete((PyObject *)self, pyname);
    if (translation) {   
      char sbuf[255];
      char *name = PyString_AsString(pyname);
      char *transname = PyString_AsString(translation);
      sprintf(sbuf, "'%s' is an (obsolete) alias for '%s'", name, transname);
      if (PyErr_Warn(PyExc_OrangeAttributeWarning, sbuf))
        return -1;
        
      res = Orange_setattr1(self, translation, args);
      Py_DECREF(translation);
      return res;
    }
    
    // Use instance's dictionary
    return Orange_setattrDictionary(self, pyname, args, warn);
    
  PyCATCH_1
}


int Orange_setattr(TPyOrange *self, PyObject *pyname, PyObject *args)
{ return Orange_setattrLow(self, pyname, args, true); }


PyObject *callbackOutput(PyObject *self, PyObject *args, PyObject *kwds,
                         char *formatname1, char *formatname2, PyTypeObject *toBase)
{ 
  PyObject *output;

  char os1[256] = "__output_";
  strcat(os1, formatname1);

  char os2[256] = "__output_";
  if (formatname2)
    strcat(os2, formatname2);

  for(PyTypeObject *type = self->ob_type;;type = type->tp_base) {
    PyObject *type_py = (PyObject *)type;

    if (PyObject_HasAttrString(type_py, os1)) {
      output = PyObject_GetAttrString(type_py, os1);
      break;
    }

    char os2[256] = "__output_";
    if (formatname2 && PyObject_HasAttrString(type_py, os2)) {
      output = PyObject_GetAttrString(type_py, os2);
      break;
    }

    if (type==toBase)
      return PYNULL;
  }

  PyObject *function = PyMethod_Function(output);
  PyObject *result;
  if (!args)
    result = PyObject_CallFunction(function, "O", self);
  else {
    PyObject *margs = PyTuple_New(1+PyTuple_Size(args));
    
    Py_INCREF(self);
    PyTuple_SetItem(margs, 0, self);
    for(Py_ssize_t i = 0, e = PyTuple_Size(args); i<e; i++) {
      PyObject *t = PyTuple_GetItem(args, i);
      Py_INCREF(t);
      PyTuple_SetItem(margs, i+1, t);
    }

    result = PyObject_Call(function, margs, kwds);
    Py_DECREF(margs);
  }

  Py_DECREF(output);
  return result;
}
  

char const *getName(TPyOrange *self)
{ static char *namebuf = NULL;

  if (namebuf) {
    delete namebuf;
    namebuf = NULL;
  }
    
  PyObject *pystr = PyString_FromString("name");
  PyObject *pyname = Orange_getattr(self, pystr);
  if (!pyname) {
    PyErr_Clear();
    return NULL;
  }

  Py_DECREF(pystr);

  if (!PyString_Check(pyname)) {
    pystr = PyObject_Repr(pyname);
    Py_DECREF(pyname);
    pyname = pystr;
  }

  const Py_ssize_t sze = PyString_Size(pyname);
  if (sze) {
    namebuf = mlnew char[sze+1];
    strcpy(namebuf, PyString_AsString(pyname));
  }
  Py_DECREF(pyname);

  return namebuf;
}


PyObject *Orange_repr(TPyOrange *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "repr", "str");
    if (result)
      return result;

    const char *tp_name = self->ob_type->tp_name + (strncmp(self->ob_type->tp_name, "orange.", 7) ? 0 : 7);
    const char *name = getName(self);
    return name ? PyString_FromFormat("%s '%s'", tp_name, name)
                : PyString_FromFormat("<%s instance at %p>", tp_name, self->ptr);
  PyCATCH
}


PyObject *Orange_str(TPyOrange *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;

    const char *tp_name = self->ob_type->tp_name + (strncmp(self->ob_type->tp_name, "orange.", 7) ? 0 : 7);
    const char *name = getName(self);
    return name ? PyString_FromFormat("%s '%s'", tp_name, name)
                : PyString_FromFormat("<%s instance at %p>", tp_name, self->ptr);
  PyCATCH
}


int Orange_nonzero(PyObject *self)
{ PyTRY
    if (self->ob_type->tp_as_sequence && self->ob_type->tp_as_sequence->sq_length)
      return self->ob_type->tp_as_sequence->sq_length(self) ? 1 : 0;
      
    if (self->ob_type->tp_as_mapping && self->ob_type->tp_as_mapping->mp_length)
      return self->ob_type->tp_as_mapping->mp_length(self) ? 1 : 0;
      
    return PyOrange_AS_Orange(self) ? 1 : 0;
  PyCATCH_1
}

 
int Orange_hash(TPyOrange *self)
{ return _Py_HashPointer(self); }


PyObject *Orange_setattr_force(TPyOrange *self, PyObject *args) PYARGS(METH_VARARGS, "(name, value) -> None") //>setattr
{ 
  PyObject *pyname, *pyvalue;
  if (!PyArg_ParseTuple(args, "OO:Orange.setattr", &pyname, &pyvalue))
    return PYNULL;
  if (!PyString_Check(pyname))
    PYERROR(PyExc_TypeError, "attribute name must be a string", PYNULL);
  if (Orange_setattrLow(self, pyname, pyvalue, false) == -1)
    return PYNULL;
  RETURN_NONE;
}


PyObject *Orange_clone(TPyOrange *self) PYARGS(METH_NOARGS, "() -> a sensibly deep copy of the object")
{
  return WrapOrange(POrange(CLONE(TOrange, ((TOrange *)self->ptr))));
}

PyObject *Orange_reference(TPyOrange *self) PYARGS(METH_NOARGS, "() -> reference; Returns unique id for an object")
{ PyTRY
    return PyInt_FromLong(long(self->ptr));
  PyCATCH
}


PyObject *Orange_typeid(TPyOrange *self) PYARGS(METH_NOARGS, "() -> int; Returns unique id for object's type")
{ PyTRY
    return PyInt_FromLong(long(&typeid(*self->ptr))); 
  PyCATCH
}


PyObject *Orange_dump(PyObject *self, PyObject *args, PyObject *kwd) PYARGS(METH_VARARGS | METH_KEYWORDS, "(formatname, ...) -> string; Prints the object into string")
{ PyTRY
    if (!args || !PyTuple_Size(args)) {
      PyErr_Format(PyExc_AttributeError, "missing arguments for '%s'.output", self->ob_type->tp_name);
      return PYNULL;
    }

    PyObject *stype = PyTuple_GetItem(args, 0);
    if (!PyString_Check(stype)) {
      PyErr_Format(PyExc_AttributeError, "invalid format argument for '%s'.output", self->ob_type->tp_name);
      return PYNULL;
    }
    char *formatname = PyString_AsString(stype);
    
    PyObject *margs = PyTuple_New(PyTuple_Size(args)-1);
    for (Py_ssize_t i = 1, e = PyTuple_Size(args); i<e; i++) {
      PyObject *t = PyTuple_GetItem(args, i);
      Py_INCREF(t);
      PyTuple_SetItem(margs, i-1, t);
    }

    PyObject *result = callbackOutput(self, margs, kwd, formatname);
    if (!result && !PyErr_Occurred())
      PyErr_Format(PyExc_AttributeError, "Class '%s' cannot be dumped as '%s'", self->ob_type->tp_name, formatname);
    
    Py_DECREF(margs);
    return result;
  PyCATCH
}


PyObject *Orange_write(PyObject *self, PyObject *args, PyObject *kwd) PYARGS(METH_VARARGS | METH_KEYWORDS, "(formatname, file, ...) -> string; Writes the object to a file")
{ PyTRY
    if (!args || PyTuple_Size(args)<2) {
      PyErr_Format(PyExc_AttributeError, "missing arguments for '%s'.output", self->ob_type->tp_name);
      return PYNULL;
    }

    PyObject *stype = PyTuple_GetItem(args, 0);
    if (!PyString_Check(stype)) {
      PyErr_Format(PyExc_AttributeError, "invalid format argument for '%s'.output", self->ob_type->tp_name);
      return PYNULL;
    }
    char *formatname = PyString_AsString(stype);
    
    PyObject *margs = PyTuple_New(PyTuple_Size(args)-2);
    for (Py_ssize_t i = 2, e = PyTuple_Size(args); i<e; i++) {
      PyObject *t = PyTuple_GetItem(args, i);
      Py_INCREF(t);
      PyTuple_SetItem(margs, i-2, t);
    }

    PyObject *result = callbackOutput(self, margs, kwd, formatname);
    Py_DECREF(margs);

    if (!result)
      return PYNULL;

    PyObject *pfile = PyTuple_GetItem(args, 1);
    if (pfile)
      if (PyFile_Check(pfile))
        Py_INCREF(pfile);
      else
        if (PyString_Check(pfile))
          pfile = PyFile_FromString(PyString_AsString(pfile), "wb");
        else
          pfile = NULL;
    
    if (!pfile) {
      PyErr_Format(PyExc_AttributeError, "invalid format argument for '%s'.output", self->ob_type->tp_name);
      Py_DECREF(result);
      return PYNULL;
    }

    int succ = PyFile_WriteObject(result, pfile, Py_PRINT_RAW);
    Py_DECREF(result);
    Py_DECREF(pfile);

    if (succ<0) {
      if (!PyErr_Occurred())
        PyErr_Format(PyExc_AttributeError, "Class '%s' cannot be written as '%s'", self->ob_type->tp_name, formatname);
      return PYNULL;
    }
    else
      RETURN_NONE;

  PyCATCH
}


#include <typeinfo>
#include <string>


bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base)
{ if (PyString_Check(obj))
    str=PyString_AsString(obj);
  else if (PyObject_TypeCheck(obj, (PyTypeObject *)const_cast<TOrangeType *>(&base)))
    str = string(getName((TPyOrange *)obj));
  else
    PYERROR(PyExc_TypeError, "invalid argument type", false);

  return true;
}


#include "cls_orange.px"
