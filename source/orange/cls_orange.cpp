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


#include <string.h>

#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"

#include "orange.hpp"
#include "cls_value.hpp"
#include "cls_example.hpp"
#include "cls_orange.hpp"

#include "externs.px"


BASED_ON(Orange, ROOT)
DATASTRUCTURE(Orange, TPyOrange, orange_dict)


POrange PyOrType_NoConstructor()
{ throw mlexception("no constructor for this type");
  return POrange();
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
    // the latter is pyxtract responsibility and pyxtract shouldn't be doubted
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

    if (!PyTuple_Size(args)) {
      Orange_init(self1, args, kwds);
      return self1;
    }

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

      // tp_call also sets the properties from kwds; no need to call tp_init before
      PyObject *self2=self1->ob_type->tp_call(self1, args, kwds);

      if (self2) {
        if (type!=self1->ob_type) {
          PyObject *ccReturn = PyObject_GetAttrString((PyObject *)type, "_AssociationRulesInducer__call_construction_type");

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



PyObject *WrapOrange(POrange obj)
{ 
  if (!obj)
    RETURN_NONE;

  PyTRY
    PyObject *res=(PyObject *)obj.counter;

    if (res->ob_type==(PyTypeObject *)&PyOrOrange_Type) {
      PyTypeObject *type = (PyTypeObject *)FindOrangeType(obj);
      if (!type) {
        PyErr_Format(PyExc_SystemError, "Orange class '%s' not exported to Python", TYPENAME(typeid(obj.getReference())));
        return PYNULL;
      }
      else
        res->ob_type = type;
    }
      
    Py_INCREF(res);
    return res;
  PyCATCH
}



int Orange_traverse(TPyOrange *self, visitproc visit, void *arg)
{ return self->ptr->traverse(visit, arg);
}


int Orange_clear(TPyOrange *self)
{ return self->ptr->dropReferences();
}



PyObject *Orange_getattr1(TPyOrange *self, PyObject *pyname);

PyObject *PyOrange__dict__(TPyOrange *self)
{ PyObject *dict = PyDict_New();
  for (const TPropertyDescription *pd = PyOrange_AS_Orange(self)->classDescription()->properties; pd->name; pd++) {
    PyObject *pyname = PyString_FromString(pd->name);
    PyObject *pyval = Orange_getattr1(self, pyname);
    if (!pyval) {
      Py_DECREF(pyname);
      Py_DECREF(dict);
      return PYNULL;
    }
    PyDict_SetItem(dict, pyname, pyval);
    Py_DECREF(pyname);
  }

  PyObject **odict = _PyObject_GetDictPtr((PyObject *)self);
  if (odict && *odict)
    PyDict_Update(dict, *odict);

  return dict;
}


PyObject *PyOrange_translateObsolete(PyObject *self, PyObject *pyname)
{ 
  char *name=PyString_AsString(pyname);
  for(TOrangeType *selftype=PyOrange_OrangeBaseClass(self->ob_type); PyOrange_CheckType((PyTypeObject *)selftype); selftype=(TOrangeType *)(selftype->ot_inherited.tp_base))
    if (selftype->ot_aliases)
      for(TAttributeAlias *aliases=selftype->ot_aliases; aliases->alias; aliases++)
        if (!strcmp(name, aliases->alias))
          return PyString_FromString(aliases->realName);
  return NULL;
}    


PyObject *Orange_getattr1(TPyOrange *self, PyObject *pyname)
// This is a complete getattr, but without translation of obsolete names.
{ PyTRY
    if (!self)
      PYERROR(PyExc_SystemError, "NULL Orange object", PYNULL);

    PyObject *res=PyObject_GenericGetAttr((PyObject *)self, pyname);
    if (res || !PyString_Check(pyname))
      return res;

    PyErr_Clear();

    char *name=PyString_AsString(pyname);

    if (strcmp(name, "__dict__") == 0)
      return PyOrange__dict__(self);
  
    // Else, try to get this as a property
    try {
      const type_info &propertyType=self->ptr->propertyType(name);

      if (propertyType==typeid(bool)) {
        bool value;
        self->ptr->getProperty(name, value);
        return Py_BuildValue("i", value ? 1 : 0);
      }

      if (propertyType==typeid(int)) {
        int value;
        self->ptr->getProperty(name, value);
        return Py_BuildValue("i", value);
      }

      if (propertyType==typeid(float)) {
        float value;
        self->ptr->getProperty(name, value);
        return Py_BuildValue("f", value);
      }

      if (propertyType==typeid(string)) {
        string value;
        self->ptr->getProperty(name, value);
        return Py_BuildValue("s", value.c_str());
      }

      if (propertyType==typeid(TValue)) {
        TValue value;
        self->ptr->getProperty(name, value);
        return Value_FromValue(value);
      }

      if (propertyType==typeid(TExample)) {
        POrange mlobj;
        self->ptr->wr_getProperty(name, mlobj);
        if (mlobj)
          return Example_FromWrappedExample(PExample(mlobj));
        else
          RETURN_NONE;
      }
      
      POrange mlobj;
      self->ptr->wr_getProperty(name, mlobj);
      return (PyObject *)WrapOrange(mlobj);
    } catch (exception err)
    {};
 
    if (!strcmp(name, "name") || !strcmp(name, "shortDescription") || !strcmp(name, "description"))
      return PyString_FromString("");

    PyErr_Format(PyExc_AttributeError, "'%s' has no attribute '%s'", self->ob_type->tp_name, name);
    return PYNULL;
  PyCATCH;
}




int Orange_setattr1(TPyOrange *self, PyObject *pyname, PyObject *args)
// This is a complete setattr, but without translation of obsolete names.
{ if (!self)
    PYERROR(PyExc_SystemError, "NULL Orange object", -1);

  char *name=PyString_AsString(pyname);
    
  const TPropertyDescription *propertyDescription;
  try {
    propertyDescription = self->ptr->propertyDescription(name);
  } catch (exception err)
  { propertyDescription = NULL; }

  PyTRY
    if (propertyDescription) {
      if (propertyDescription->readOnly) {
        /* Property might be marked as readOnly, but have a specialized set function.
           The following code is pasted from PyObject_GenericSetAttr.
           If I'd call it here and the attribute is really read-only, PyObject_GenericSetAttr
           would blatantly store it in the dictionary. */
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
            self->ptr->setProperty(name, value!=0);
          else
            self->ptr->setProperty(name, value);
          return 0;
        }

        if (propertyType==typeid(float)) {
          float value;
          if (!PyArg_Parse(args, "f", &value)) {
            PyErr_Format(PyExc_TypeError, "invalid parameter type for %s.%s', (float expected)", self->ob_type->tp_name, name);
            return -1;
          }
          self->ptr->setProperty(name, value);
          return 0;
        }

        if (propertyType==typeid(string)) {
          char *value;
          if (!PyArg_Parse(args, "s", &value)) {
            PyErr_Format(PyExc_TypeError, "invalid parameter type for %s.%s', (string expected)", self->ob_type->tp_name, name);
            return -1;
          }
          self->ptr->setProperty(name, string(value));
          return 0;
        }

        if (propertyType==typeid(TValue)) {
          TValue value;
          if (!convertFromPython(args, value))
            return -1;
          self->ptr->setProperty(name, value);
          return 0;
        }

        if (propertyType==typeid(TExample)) {
          if (args==Py_None) {
            self->ptr->wr_setProperty(name, POrange());
            return 0;
          }
          else {
            if (PyOrExample_Check(args)) {
              PyErr_Format(PyExc_TypeError, "invalid parameter type for '%s.%s', (expected 'Example', got '%s')", self->ob_type->tp_name, name, args->ob_type->tp_name);
              return -1;
            }
            self->ptr->wr_setProperty(name, POrange(PyExample_AS_Example(args)));
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
            self->ptr->wr_setProperty(name, POrange());
            return 0;
          }

          // User might have supplied the correct object
          if (PyObject_TypeCheck(args, propertyPyType)) {
            self->ptr->wr_setProperty(name, PyOrange_AS_Orange((TPyOrange *)args));
            return 0;
          }

          // User might have supplied parameters from which we can construct the object
          if (propertyPyType->tp_new) {
            PyObject *emptyDict = PyDict_New();
            PyObject *targs;
            if (PyTuple_Check(args)) {
              targs = args;
              Py_INCREF(targs);
            }
            else
              targs = Py_BuildValue("(O)", args);

            PyObject *obj = propertyPyType->tp_new(propertyPyType, targs, emptyDict);

            // If this failed, maybe the constructor actually expected a tuple...
            if (!obj && PyTuple_Check(args)) {
              PyErr_Clear();
              Py_DECREF(targs);
              targs = Py_BuildValue("(O)", args);
              obj = propertyPyType->tp_new(propertyPyType, targs, emptyDict);
            }

            if (obj) {
              if (   propertyPyType->tp_init != NULL
		              && propertyPyType->tp_init(obj, targs, emptyDict) < 0) {
			          Py_DECREF(obj);
			          obj = NULL;
		          }

              Py_DECREF(emptyDict);
              Py_DECREF(targs);

              if (obj) {
                self->ptr->wr_setProperty(name, PyOrange_AS_Orange((TPyOrange *)obj));
                Py_DECREF(obj);
                return 0;
              }
            }
            else {
              Py_DECREF(emptyDict);
              Py_DECREF(targs);
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
    }

  PyCATCH_1;
  // this reports only errors that occur when setting built-in properties

  // setting 'name', 'shortDescription', 'description' or setting any attributes to types that are
  // *derived* from Orange types is OK and without warning
  if (strcmp(name, "name") && strcmp(name, "shortDescription") && strcmp(name, "description") && PyOrange_CheckType(self->ob_type)) {
    char sbuf[255];
    sprintf(sbuf, "'%s' is not a builtin attribute of '%s'", name, self->ob_type->tp_name);
    if (PyErr_Warn(PyExc_OrangeAttributeWarning, sbuf))
      return -1;
  }

  return PyObject_GenericSetAttr((PyObject *)self, pyname, args);
}





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

  Py_XDECREF(self->orange_dict);
  self->ob_type->tp_free((PyObject *)self);
}



PyObject *Orange_getattr(TPyOrange *self, PyObject *name)
// This calls getattr1; first with the given, than with the translated name
{ 
  PyTRY
    PyObject *res=Orange_getattr1(self, name);
    if (!res) {
      PyObject *translation=PyOrange_translateObsolete((PyObject *)self, name);
      if (translation) {
        PyErr_Clear();
        res = Orange_getattr1(self, translation);
        Py_DECREF(translation);
      }
    }

    return res;
  PyCATCH
}


int Orange_setattr(TPyOrange *self, PyObject *name, PyObject *args)
// This calls setattr1; first with the given, than with the translated name
{ PyTRY
    int res=Orange_setattr1(self, name, args);
    if (res<0) {
      PyObject *translation=PyOrange_translateObsolete((PyObject *)self, name);
      if (translation) {
        PyErr_Clear();
        res=Orange_setattr1(self, translation, args);
        Py_DECREF(translation);
      }
    }
    return res;
  PyCATCH_1
}



PyObject *callbackOutput(PyObject *self, PyObject *args, PyObject *kwds,
                         char *formatname1, char *formatname2 = NULL,
                         PyTypeObject *toBase = (PyTypeObject *)&PyOrOrange_Type)
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
    for(int i = 0, e = PyTuple_Size(args); i<e; i++) {
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
  Py_DECREF(pystr);

  if (!PyString_Check(pyname)) {
    pystr = PyObject_Repr(pyname);
    Py_DECREF(pyname);
    pyname = pystr;
  }

  const int sze = PyString_Size(pyname);
  if (sze) {
    namebuf = mlnew char[PyString_Size(pyname)+1];
    strcpy(namebuf, PyString_AsString(pyname));
    Py_DECREF(pyname);
  }

  return namebuf;
}


PyObject *Orange_repr(TPyOrange *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "repr", "str");
    if (result)
      return result;

    const char *name = getName(self);
    return name ? PyString_FromFormat("%s '%s'", self->ob_type->tp_name, name)
                : PyString_FromFormat("<%s instance at %p>", self->ob_type->tp_name, self->ptr);
  PyCATCH
}


PyObject *Orange_str(TPyOrange *self)
{ PyTRY
    PyObject *result = callbackOutput((PyObject *)self, NULL, NULL, "str", "repr");
    if (result)
      return result;

    const char *name = getName(self);
    return name ? PyString_FromFormat("%s '%s'", self->ob_type->tp_name, name)
                : PyString_FromFormat("<%s instance at %p>", self->ob_type->tp_name, self->ptr);
  PyCATCH
}


int Orange_nonzero(PyObject *self)
{ PyTRY
    return PyOrange_AS_Orange(self) ? 1 : 0;
  PyCATCH_1
}

 
int Orange_hash(TPyOrange *self)
{ return _Py_HashPointer(self); }


PyObject *Orange_enableattributes(PyObject *, PyObject *)
{ RETURN_NONE; }


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
    for (int i = 1, e = PyTuple_Size(args); i<e; i++) {
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
    for (int i = 2, e = PyTuple_Size(args); i<e; i++) {
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
