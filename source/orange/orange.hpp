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


#ifndef __ORANGE_HPP
#define __ORANGE_HPP

#include "../pyxtract/pyxtract_macros.hpp"

#include "c2py.hpp"
#include "root.hpp"

ORANGE_API PyObject *PyOrType_GenericAbstract(PyTypeObject *thistype, PyTypeObject *type, PyObject *args, PyObject *kwds);
ORANGE_API PyObject *PyOrType_GenericNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
ORANGE_API PyObject *PyOrType_GenericNamedNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
ORANGE_API PyObject *PyOrType_GenericCallableNew(PyTypeObject *type, PyObject *args, PyObject *kwds);
ORANGE_API POrange PyOrType_NoConstructor();


/* Use WrapOrange on POrange, and WrapNewOrange on freshly constructed instances of
   TOrange * that were never wrapped before (do not use WrapNewOrange on objects that
   were already wrapped, since it will attempt to change the object's ob_type */

/* This function is called by WrapOrange and WrapNewOrange. Never call it directly (unless you know why). */
ORANGE_API PyObject *WrapWrappedOrange(TWrapped *obj);

inline PyObject *WrapOrange(const TWrapper &o)
{ return WrapWrappedOrange(o.counter ? const_cast<TWrapped *>(o.counter->ptr) : NULL); }

inline PyObject *WrapNewOrange(TOrange *o, PyTypeObject *type)
{ 
  if (!o) {
    PyErr_Format(PyExc_SystemError, "Constructor for '%s' failed", type->tp_name);
    return NULL;
  }

  return WrapWrappedOrange(POrange(o, type).getUnwrappedPtr());
}
//#define WrapNewOrange(o, type) WrapWrappedOrange(POrange((TOrange *)(checkConstructed((o), type)), type).getUnwrappedPtr())



extern void *pNotConstructed;

inline void *checkConstructed(void *p, PyTypeObject *type)
{ if (!p) 
  return p;
}

/* Casting operators that are used as functions;
   they raise C++ exceptions if something goes wrong */

inline void *guarded_cast(TOrange *op, void *res, const char *name)
{ 
  if (!op) raiseError("null pointer to '%s'", name);
  else if (!res) raiseError("cannot cast from '%s' to '%s'", typeid(op).name(), name);
  return res;
}


#define PyOrange_AS(type, op) (*(type *)guarded_cast(((TPyOrange *)op)->ptr, dynamic_cast<type *>(((TPyOrange *)op)->ptr), typeid(type).name()))
#define SELF_AS(type)         (*dynamic_cast<type *>(((TPyOrange *)self)->ptr))

// Just force GCPtr to use the correct constructor
#define PyOrange_AS_Orange(op) (GCPtr<TOrange>((TPyOrange *)op, true))

/* Casting operators that define a new variable 
   XXX Why are these so different from above? We couldn't use a similar mechanism or what? */

#define NAME_CAST_TO_err(type, aname, obj, errreturn) \
  type *obj = CAST(PyOrange_AS_Orange(aname), type); \
  if (!obj) {\
    if (aname && ((TPyOrange *)aname)->ptr) \
      PyErr_Format(PyExc_TypeError, "invalid object type (expected '%s', got '%s')", TYPENAME(typeid(type)), TYPENAME(typeid(*((TPyOrange *)(aname))->ptr))); \
    else \
      PyErr_Format(PyExc_TypeError, "invalid object type (expected '%s', got nothing)", TYPENAME(typeid(type))); \
    return errreturn; \
  }

#define NAME_CAST_TO(type, name, obj)     NAME_CAST_TO_err(type, name, obj, PYNULL)
#define CAST_TO_err(type, obj, errreturn) NAME_CAST_TO_err(type, self, obj, errreturn)
#define CAST_TO(type, obj)                NAME_CAST_TO_err(type, self, obj, PYNULL)



extern ORANGE_API PyObject *orangeModule;

extern ORANGE_API PyObject *PyExc_OrangeKernel,
                           *PyExc_OrangeAttributeWarning,
                           *PyExc_OrangeWarning,
                           *PyExc_OrangeCompatibilityWarning,
                           *PyExc_OrangeKernelWarning;


#define PyCATCH_r(e) PyCATCH_r_et(e,PyExc_OrangeKernel)

void raiseWarning(bool, const char *s);
bool raiseWarning(PyObject *warnType, const char *s, ...);
bool raiseCompatibilityWarning(const char *s, ...);
/*
void raiseError(PyObject *excType, const char *s, ...);
*/


typedef struct {
  char *alias, *realName;
} TAttributeAlias;


int ORANGE_API PyOr_noConversion(PyObject *obj, void *ptr);


typedef POrange (*defaultconstrproc)(PyTypeObject *);
typedef int (*argconverter)(PyObject *, void *);


#ifdef _MSC_VER
  #pragma warning (push)
  #pragma warning (disable : 4512) // assigment operator could not be generated (occurs below, due to references)
  #undef min
  #undef max
#endif

class ORANGE_API TOrangeType { 
public:

  PyTypeObject       ot_inherited;
  const type_info   &ot_classinfo;
  defaultconstrproc  ot_defaultconstruct;
  char             **ot_constructorkeywords;
  bool               ot_constructorAllowsEmptyArgs;
  char             **ot_recognizedattributes;
  TAttributeAlias   *ot_aliases;
  argconverter       ot_converter;
  argconverter       ot_nconverter;

  /*TOrangeType()
  : ot_classinfo(typeid(TOrangeType))
  { raiseErrorWho("TOrangeType", "Internal error: invalid constructor called"); }
*/
  TOrangeType(const PyTypeObject &inh, const type_info &cinf, defaultconstrproc dc,
              argconverter otc, argconverter otcn,
              char **ck = NULL, bool caea = false, char **ra = NULL, TAttributeAlias *ali = NULL
             )
   : ot_inherited(inh),
     ot_classinfo(cinf),
     ot_defaultconstruct(dc),
     ot_constructorkeywords(ck),
     ot_constructorAllowsEmptyArgs(caea),
     ot_recognizedattributes(ra),
     ot_aliases(ali),
     ot_converter(otc),
     ot_nconverter(otcn)
   {}
};

#ifdef _MSC_VER
  #pragma warning (pop)
#endif


ORANGE_API void addClassList(TOrangeType **);

ORANGE_API TOrangeType *FindOrangeType(const type_info &);

// Checks whether the object (or type) is one of orange's types
ORANGE_API bool PyOrange_CheckType(PyTypeObject *);

// Ascends the hierarchy until it comes to a class that is from orange's hierarchy
TOrangeType *PyOrange_OrangeBaseClass(PyTypeObject *);

ORANGE_API bool SetAttr_FromDict(PyObject *self, PyObject *dict, bool fromInit = false);

#define NO_KEYWORDS { if (!((TPyOrange *)self)->call_constructed && keywords && PyDict_Size(keywords)) PYERROR(PyExc_AttributeError, "this function accepts no keyword arguments", PYNULL); }

PyObject *yieldNoPickleError(PyObject *self, PyObject *);


// Returns a borrowed reference!
inline PyObject *getExportedFunction(const char *func)
{ return PyDict_GetItemString(PyModule_GetDict(orangeModule), func); }

// Returns a borrowed reference!
inline PyObject *getExportedFunction(PyObject *module, const char *func)
{ return PyDict_GetItemString(PyModule_GetDict(module), func); }


typedef struct {
  const char *name;
  const long value;
} TNamedConstantsDef;

typedef struct {
  char *name;
  PyTypeObject *type;
} TNamedConstantRecord;

ORANGE_API PyObject *unpickleConstant(TNamedConstantRecord const *, PyObject *args);

PyObject *stringFromList(PyObject *self, TNamedConstantsDef const *ncs);

#endif
