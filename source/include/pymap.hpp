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


#ifndef __PYMAP_HPP
#define __PYMAP_HPP

#include "Python.h"
#include <map> 
#include "garbage.hpp"
using namespace std;

template<class T, class V, class U=GCPtrNML<map<T, V> > >
class TPyMap {
public:
  // Python's object head
  PyObject_HEAD

  // Definition of a wrapped vector
  typedef U PMap; //GCPtr<map<T, U> > PMap;

  // Definition of a function type for conversion of Python's object into T
  typedef bool (*TConvertFromPythonFuncT)(PyObject *, T &);
  typedef bool (*TConvertFromPythonFuncV)(PyObject *, V &);

  // The wrapped map
  PMap pymap;

  // A function for conversion from Python's object into T and V;
  // a default exists, but each object can have its specialized function
  // (this is a kind of virtuality but on the object rather than the class level)
  TConvertFromPythonFuncT convertFromPythonT;
  TConvertFromPythonFuncV convertFromPythonV;


  /* *****************************************************************/ 
  /* *** Constructors and destructors                              ***/

  // Constructs an empty map
  TPyMap (TConvertFromPythonFuncT cfpt=NULL, TConvertFromPythonFuncV cfpv=NULL)
    : ob_type(PyMap_Type), pymap(mlnew map<T,V>()),
      convertFromPythonT(cfpt ? cfpt : (TConvertFromPythonFuncT)(::convertFromPython)),
      convertFromPythonV(cfpv ? cfpv : (TConvertFromPythonFuncV)(::convertFromPython))
    { _Py_NewReference((PyObject *)this); }

  // Wrapps an existing map
  TPyMap(PMap wv, TConvertFromPythonFuncT cfpt=NULL, TConvertFromPythonFuncV cfpv=NULL)
    : ob_type(PyMap_Type), pymap(wv),
      convertFromPythonT(cfpt ? cfpt : (TConvertFromPythonFuncT)(::convertFromPython)),
      convertFromPythonV(cfpv ? cfpv : (TConvertFromPythonFuncV)(::convertFromPython))
    { _Py_NewReference((PyObject *)this); }


  // Constructs a map from Python arguments
  TPyMap(PyObject *args, TConvertFromPythonFuncT cfpt=NULL, TConvertFromPythonFuncV cfpv=NULL)
    : ob_type(PyMap_Type), pymap(NULL),
      convertFromPythonT(cfpt ? cfpt : (TConvertFromPythonFuncT)(::convertFromPython)),
      convertFromPythonV(cfpv ? cfpv : (TConvertFromPythonFuncV)(::convertFromPython))
    { TPyVector_readMap(pymap, args, convertFromPythonT, convertFromPythonV); }

  // Deallocation caused by Python
  void dealloc()
  { /* !!!!! IS THIS A MEMORY LEAK OR NOT?!?!?! !!!!!! */}


  /* *****************************************************************/ 
  /* *** List related methods as member functions                  ***/

  PyObject *keys(PyObject *)
  { PyObject *list=PyList_New(pymap ? pymap->size() : 0);
    int i=0;
    if (pymap)
      for(typename map<T, V>::const_iterator ii(pymap->begin()), ei(pymap->end()); ii!=ei; ii++)
        PyList_SetItem(list, i++, convertToPython((*ii).first));
    return list;
  }


  PyObject *values(PyObject *)
  { PyObject *list=PyList_New(pymap ? pymap->size() : 0);
    int i=0;
    if (pymap)
      for(typename map<T, V>::const_iterator ii(pymap->begin()), ei(pymap->end()); ii!=ei; ii++)
        PyList_SetItem(list, i++, convertToPython((*ii).second));
    return list;
  }


  PyObject *items(PyObject *)
  { PyObject *list=PyList_New(pymap ? pymap->size() : 0);
    int i=0;
    if (pymap)
      for(typename map<T, V>::const_iterator ii(pymap->begin()), ei(pymap->end()); ii!=ei; ii++)
        PyList_SetItem(list, i++, Py_BuildValue("NN", convertToPython((*ii).first), convertToPython((*ii).second)));
    return list;
  }


  string PyMap2string() const
  { string res("{");
    if (pymap)
      for(typename map<T, V>::const_iterator ei(pymap->begin()), ee(pymap->end()); ei!=ee; ei++) {
        if (res.length()>1) res+=", ";
        res+= convertToString((*ei).first) + ": " + convertToString((*ei).second);
      }
    return res+"}";
  }


  int length() const
  { return pymap->size(); }



  PyObject *getItem(PyObject *pykey)
  { if (!pymap)
      PYERROR(PyExc_KeyError, "invalid key", NULL);
    T key;
    if (!convertFromPython(pykey, key))
	    PYERROR(PyExc_KeyError, "invalid key", NULL);
    typename map<T, V>::const_iterator fi=pymap->find(key);
    if (fi==pymap->end())  
      PYERROR(PyExc_KeyError, "invalid key", NULL);

    return convertToPython((*fi).second);
  }


  int setItem(PyObject *pykey, PyObject *pyvalue)
  { if (!pymap)
      pymap=mlnew map<T, V>();

    T key;
    V value;
    if (!convertFromPython(pykey, key))
	    PYERROR(PyExc_KeyError, "invalid key", -1);
    if (!convertFromPython(pyvalue, value))
	    PYERROR(PyExc_KeyError, "invalid value", -1);

    pymap->operator[](key)=value;
    return 0;
  }


  int cmp(const TPyMap<T,V,U> *another) const
  { if (!pymap || !another->pymap)
      return 1;

    for(typename map<T,V>::const_iterator  b1(pymap->begin()), e1(pymap->end()),
                                           b2(another->pymap->begin()), e2(another->pymap->end());
        (b1!=e1) && (b2!=e2); b1++, b2++)
      if (*b1!=*b2) return 1;
    return 0;
  }


  /* *****************************************************************/ 
  /* *** List related methods as static functions                  ***/
  /*     to be called directly from Python (they call member functions) */

  static PyObject *static_keys(TPyMap<T,V,U> *self, PyObject *args)
  { return self->keys(args); }

  static PyObject *static_values(TPyMap<T,V,U> *self, PyObject *args)
  { return self->values(args); }

  static PyObject *static_items(TPyMap<T,V,U> *self, PyObject *args)
  { return self->items(args); }

  static int static_length(TPyMap<T,V,U> *self)
  { return self->length(); }

  static PyObject *static_getItem(TPyMap<T,V,U> *self, PyObject *pykey)
  { return self->getItem(pykey); }

  static int static_setItem(TPyMap<T,V,U> *self, PyObject *pykey, PyObject *pyvalue)
  { return self->setItem(pykey, pyvalue); }

  static int static_cmp(TPyMap<T,V,U> *self, TPyMap<T,V,U> *another)
  { return self->cmp(another); }
  

  // The following methods have no member counterparts

  static PyObject *static_getattr(PyObject *self, char *name)
  { return Py_FindMethod(PyMap_Methods, self, name); }

  static int static_print(TPyMap<T,V,U> *self, FILE *fp, int flags)
  { fprintf(fp, self->PyMap2string().c_str()); 
    return 0;
  }

  static PyObject *static_repr(TPyMap<T,V,U> *self)
  { return PyString_FromString(self->PyMap2string().c_str()); }

  static void static_dealloc(TPyMap<T,V,U> *self)
  { self->dealloc();
    mldelete self;
  }


  /* *****************************************************************/ 
  /* *** Type and methods definitions, common for all vectors<T>   ***/

  static PyTypeObject *PyMap_Type;
  static PyMethodDef *PyMap_Methods;

  // A function that once for all creates the type definition structures and the methods' array
  static PyTypeObject *defineMapType(char *name=NULL)
  {

    static PyMethodDef methods[4] = { 
      {"keys", (binaryfunc)static_keys, 1},
      {"values", (binaryfunc)static_values, 1},
      {"items", (binaryfunc)static_items, 1},
      {NULL, NULL}
    };

    static PyMappingMethods TPyMap_as_mapping={
      (inquiry)       static_length,
      (binaryfunc)    static_getItem,
      (objobjargproc) static_setItem,
    };

    if (!name) {
      name=mlnew char[10+strlen(typeid(T).name())];
      sprintf(name, "map<%s>", typeid(T).name());
    }

    static PyTypeObject type={
      PyObject_HEAD_INIT(&PyType_Type)
      0,
      name,
      sizeof(TPyMap<T,V,U>),
      0,

      (destructor)   static_dealloc,
      0, //(printfunc)    static_print,
      (getattrfunc)  static_getattr,
      0,
      (cmpfunc)      static_cmp,
      (reprfunc)     static_repr,
  
      0,
      0,
      &TPyMap_as_mapping,

      0, 0,
      (reprfunc)     static_repr
    };

    PyMap_Methods=methods;
  
    return &type;
  }


  /* *****************************************************************/ 
  /* *** Methods callable from C++                                 ***/

  // Checks whether the given vector is of this Python's type
  static inline bool check(PyObject *self)
    { return self->ob_type==PyMap_Type; }


  // This is not related to PyVector<T,U> but to vector<T>;
  //   it reads a list in the same format as for constructor (actully, the constructor
  //   calls this function), but places the result into a vector<T>. This function is
  //   useful for setting an attribute (field) of C++ object which is otherwise returned
  //   as PyVector<T,U>.
  static bool readMap(PMap &pymap, PyObject *args, TConvertFromPythonFuncT convertT=NULL, TConvertFromPythonFuncV convertV=NULL)
  { 
    if (!args) {
      pymap=PMap(mlnew map<T,V>());
      return true;
    }

    if (PyDict_Check(args)) {
      pymap=PMap(mlnew map<T,V>());
      PyObject *pykey, *pyvalue;
      int pos=0;
      while (PyDict_Next(args, &pos, &pykey, &pyvalue)) {
        T key;
        V value;
        if (! (convertT ? convertT(pykey, key) : convertFromPython(pykey, key))) {
          pymap=mlnew map<T, V>();
	        PYERROR(PyExc_KeyError, "invalid key", false);
        }
        if (! (convertV ? convertV(pyvalue, value) : convertFromPython(pyvalue, value))) {
          pymap=mlnew map<T, V>();
	        PYERROR(PyExc_KeyError, "invalid value", false);
        }

        pymap->operator[](key)=value;
      }
    }

    else if (check(args))
      pymap=((TPyMap<T,V,U> *)(args))->pymap;

    else return false;
       
    return true;
  }

};

// A macro that returns a map<T> of PyMap<T>. No type checking
#define PyMap_AS_Map(obj,T,V) ((TPyMap<T,V> *)(obj))->pymap

// A macro that defines the static fields of PyMap<T,V>
// and calls the defineMapType to initialize them.
#define DefineMapType(T,V,name) \
  PyMethodDef *TPyMap<T,V>::PyMap_Methods; \
  PyTypeObject *TPyMap<T,V>::PyMap_Type=TPyMap<T,V>::defineMapType(name);

// A macro that defines the static fields of PyMap<T,V,U>
// and calls the define vector to initialize them.
#define DefineMapTypeU(T,V,U,name) \
  PyMethodDef *TPyMap<T,V,U>::PyMap_Methods; \
  PyTypeObject *TPyMap<T,V,U>::PyMap_Type=TPyMap<T,V,U>::defineMapType(name);


// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX

#endif

