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

#include "cls_value.hpp"
#include "cls_example.hpp"
#include "cls_orange.hpp"

#include "externs.px"


BASED_ON(Orange, ROOT)
DATASTRUCTURE(Orange, TPyOrange, orange_dict)


class TPyOrange_DictProxy : public PyDictObject {
public:
  TPyOrange *backlink;
};


POrange PyOrType_NoConstructor()
{ throw mlexception("no constructor for this type");
  return POrange();
}


TOrangeType *FindOrangeType(const type_info &tinfo)
{ TOrangeType **orty=orangeClasses;
  while (*orty && ((*orty)->ot_classinfo!=tinfo))
    orty++;

  return *orty;
}

bool PyOrange_CheckType(PyTypeObject *pytype)
{ TOrangeType *type=(TOrangeType *)pytype;
  for(TOrangeType **orty=orangeClasses; *orty; orty++)
    if (*orty==type)
      return true;
  return false;
}


// Ascends the hierarchy until it comes to a class that is from orange's hierarchy
TOrangeType *PyOrange_OrangeBaseClass(PyTypeObject *pytype)
{ while (pytype && !PyOrange_CheckType(pytype))
    pytype=pytype->tp_base;
  return (TOrangeType *)pytype;
}


bool SetAttr_FromDict(PyObject *self, PyObject *dict, bool fromInit)
{
  if (dict) {
    int pos = 0;
    PyObject *key, *value;
    char **kc = fromInit ? ((TOrangeType *)(self->ob_type))->ot_constructorkeywords : NULL;
    while (PyDict_Next(dict, &pos, &key, &value)) {
      if (kc) {
        char *kw = PyString_AsString(key);
        char **akc;
        for (akc = kc; *akc && strcmp(*akc, kw); akc++);
        if (*akc)
          continue;
      }
      if (PyObject_SetAttr(self, key, value)<0)
        return false;
    }
  }
  return true;
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
{
  if (self->orange_dict) {
    int err = visit(self->orange_dict, arg);
    if (err)
      return err;
  }

  return self->ptr->traverse(visit, arg);
}


int Orange_clear(TPyOrange *self)
{ 
  return self->ptr->dropReferences();
}


PyObject *PyOrange_DictProxy_New(TPyOrange *);
PyObject *Orange_getattr1(TPyOrange *self, PyObject *pyname);

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


PyObject *PyOrange_translateObsolete(PyObject *self, PyObject *pyname)
{ 
  char *name=PyString_AsString(pyname);
  for(TOrangeType *selftype = PyOrange_OrangeBaseClass(self->ob_type); PyOrange_CheckType((PyTypeObject *)selftype); selftype=(TOrangeType *)(selftype->ot_inherited.tp_base))
    if (selftype->ot_aliases)
      for(TAttributeAlias *aliases=selftype->ot_aliases; aliases->alias; aliases++)
        if (!strcmp(name, aliases->alias))
          return PyString_FromString(aliases->realName);
  return NULL;
}    


PyObject *Orange_getattr1(TPyOrange *self, const char *name)
// This is a getattr, without translation of obsolete names and without looking into the associated dictionary
{ PyTRY
    if (!self)
      PYERROR(PyExc_SystemError, "NULL Orange object", PYNULL);

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



int Orange_setattr1(TPyOrange *self, PyObject *pyname, PyObject *args)
// This is a complete setattr, but without translation of obsolete names.
{ if (!self)
    PYERROR(PyExc_SystemError, "NULL Orange object", -1);

  char *name=PyString_AsString(pyname);
  const TPropertyDescription *propertyDescription = self->ptr->propertyDescription(name, true);

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

            PyObject *obj = NULL;
            try {
              obj = propertyPyType->tp_new(propertyPyType, targs, emptyDict);
            }
            catch (...) {
              // do nothing; if it failed, the user probably didn't mean it
            }

            // If this failed, maybe the constructor actually expected a tuple...
            if (!obj && PyTuple_Check(args)) {
              PyErr_Clear();
              Py_DECREF(targs);
              targs = Py_BuildValue("(O)", args);
              try { obj = propertyPyType->tp_new(propertyPyType, targs, emptyDict); }
              catch (...) {}
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
                bool success = true;
                try {
                  self->ptr->wr_setProperty(name, PyOrange_AS_Orange((TPyOrange *)obj));
                }
                catch (...) {
                  success = false;
                }
                Py_DECREF(obj);
                if (success)
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
    if (!res) {
      PyObject *translation = PyOrange_translateObsolete((PyObject *)self, name);
      if (translation) {
        PyErr_Clear();
        res = Orange_getattr1(self, translation);
        Py_DECREF(translation);
      }
    }

    return res;
  PyCATCH
}

extern PyTypeObject PyOrange_DictProxy_Type;

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

    char *name = PyString_AsString(pyname);
    if (args) {
      /* Issue a warning unless name is 'name', 'shortDescription', 'description'
         or the instance's class is defined in Python and merely inherits this
         method from Orange class) */
      if (warn
          && strcmp(name, "name") && strcmp(name, "shortDescription") && strcmp(name, "description")
          && PyOrange_CheckType(self->ob_type)) {
        char sbuf[255];
        sprintf(sbuf, "'%s' is not a builtin attribute of '%s'", name, self->ob_type->tp_name);
        if (PyErr_Warn(PyExc_OrangeAttributeWarning, sbuf))
          return -1;
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

  const int sze = PyString_Size(pyname);
  if (sze) {
    namebuf = mlnew char[PyString_Size(pyname)+1];
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
       
      PyList_Append(items, Py_BuildValue("sN", pd->name, pyattr));
    }

  return items;
}



static PyObject *PyOrange_DictProxy_update(TPyOrange_DictProxy *mp, PyObject *seq2)
{
  PyObject *key, *value;
  int pos = 0;

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
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |	Py_TPFLAGS_BASETYPE,
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
	PyObject_GC_Del,
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
	int di_used;
	int di_pos;
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
	PyObject_SelfIter,			/* tp_iter */
	(iternextfunc)PyOrange_DictProxyIter_iternext,	/* tp_iternext */
	0,					/* tp_methods */
	0,					/* tp_members */
	0,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
};

#include "cls_orange.px"
