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


#ifndef __GARBAGE_HPP
#define __GARBAGE_HPP

/*

On wrapping pointers vs. wrapping references 

Wrapping references can speed up save execution time since stack allocation is
somewhat faster than heap allocations


Wrapping references is allowed, but risky. The corresponding constructor
is declared as 'explicit' to make sure that it is not called in error.

Sending wrapped reference is OK if the called function won't store it.
Example of such function is operator() of TMeasure methods that uses
PDistribution only to assess the quality of the attribute. But even in
such cases one must be cautious: what if a more complex measure, such
as ReliefF, stores something to speed up the next computation?
Because of such assumptions, it is recommended that someone that
'unexpectedly' stores something checks that the stored item is not a
wrapped reference using PyObject_IsReference macro.

In addition, you may wrap a reference if you know for sure that the
object receiving the wrapper will die before the wrapped object.

.
*/

#include <typeinfo>
#include <stdio.h>
#include <Python.h>
using namespace std;

// Not the most appropriate, but surely the most suitable place:
#include "px/orange_globals.hpp"

#define WRAPPER ORANGE_WRAPPER
#define VWRAPPER ORANGE_VWRAPPER

#ifdef _MSC_VER
  #include <crtdbg.h>
  #pragma warning (disable : 4231 4660 4661 4786 4114 4018 4267 4244 4702 4710 4290 4251 4275)
  #pragma warning (disable : 4786 4114 4018 4267 4127)
  #define TYPENAME(x) (x).name()+7

#else
  #include <assert.h>
  #define _ASSERT assert
  char *demangle(const type_info &type);
  #define TYPENAME(x) demangle(x)+1
#endif

#define mlnew new
#define mldelete delete

class TOrangeType;
extern ORANGE_API PyTypeObject PyOrNonOrange_Type;
extern ORANGE_API TOrangeType PyOrOrange_Type;

class TWrapped;

class TGCCounter {
public:
	PyObject_HEAD
  TWrapped *ptr;
  PyObject *orange_dict;
  bool call_constructed, is_reference;

  inline void getRef()
  { Py_INCREF(this); }

  inline void freeRef()
  { Py_DECREF(this); }
};


class ORANGE_API TWrapped {
public:
  TGCCounter *myWrapper;

  TWrapped()
  : myWrapper(NULL)
  {}

  virtual ~TWrapped()
  {}
};


/* Checks whether the object is gc-tracked or not.
   This way, we distinguish between wrapped pointers (which are)
   and wrapped references (which are not wrapped).
   The latter don't get freed. */

#define PyObject_IsPointer(o) (((PyGC_Head *)(o)-1)->gc.gc_next != NULL)
#define PyObject_IsReference(o) (((PyGC_Head *)(o)-1)->gc.gc_next == NULL)
  
typedef TGCCounter TPyOrange;

#define VISIT(obj) { int res=visit((PyObject *)(obj), arg); if (res) return res; }

class ORANGE_API TWrapper {
public:
  TGCCounter *counter;

  inline void init() // used when GCPtr is allocated in C code and constructor is not called
  { counter = NULL; }

  inline TWrapper()
  : counter(NULL)
  {}


  inline TWrapper(TGCCounter *ac)
  : counter(ac)
  {
    if (counter)
      counter->getRef();
  }


  inline operator bool() const
  { return (counter!=NULL); }


  ~TWrapper()
  { if (counter)
      counter->freeRef();
  }
};


template<class T>
class GCPtr : public TWrapper {
public:

  inline GCPtr() 
  {};


  inline GCPtr(TGCCounter *acounter, bool)  // Used by PyOrange_AS_Orange
  : TWrapper(acounter)
  {}


  inline GCPtr(T *ptr)
  { 
    if (ptr)
      if (((TWrapped *)ptr)->myWrapper) {
        counter = ((TWrapped *)ptr)->myWrapper;
        counter->getRef();
      }
      else {
        counter = PyObject_GC_New(TGCCounter, (PyTypeObject *)&PyOrOrange_Type);
        counter->orange_dict = NULL;
        counter->call_constructed = false;
        counter->is_reference = false;
        counter->ptr = (TWrapped *)ptr;
        ((TWrapped *)ptr)->myWrapper = counter;
        PyObject_GC_Track(counter);
      }
  }


  inline GCPtr(T *ptr, PyTypeObject *type)
  { 
    if (ptr) {
      counter = (TGCCounter *)type->tp_alloc(type, 0);
      counter->orange_dict = NULL;
      counter->call_constructed = false;
      counter->is_reference = false;
      counter->ptr = (TWrapped *)ptr;
      _ASSERT(!((TWrapped *)ptr)->myWrapper);
      ((TWrapped *)ptr)->myWrapper = counter;
    }
  }


  inline explicit GCPtr(T &ptr) 
  { 
    counter = PyObject_GC_New(TGCCounter, (PyTypeObject *)&PyOrOrange_Type);
    counter->orange_dict = NULL;
    counter->call_constructed = false;
    counter->is_reference = true; // this should never be deleted
    counter->ptr = (TWrapped *)&ptr;
    /* No calling PyObject_GC_Track(counter);
       These objects cannot be freed; if they participate in a cycle, the cycle
       will disappear as soon as they die (and these objects die soon).
       We even distinguish between wrapped pointers and reference by checking
       whether they are gc-tracked or not. */
  }


  template<class U>
  GCPtr(const GCPtr<U> &other)
  : TWrapper(other.counter)
  { if (counter) {
      if (!dynamic_cast<T *>(other.counter->ptr))
        raiseError("bad cast from %s to %s", typeid(U).name(), typeid(T).name());
    }
  }


/*  GCPtr(const TWrapper &other)
  : TWrapper(other.counter)
  { 
    if (counter && !dynamic_cast<T *>(other.counter->ptr))
      raiseError("bad cast from %s to %s", typeid(other).name(), typeid(T).name());
  }
*/

  GCPtr(const GCPtr<T> &other)
  : TWrapper(other.counter)
  {}


  GCPtr<T> & operator =(const GCPtr<T> &other)
  { 
    if (other.counter)
      other.counter->getRef();
    if (counter)
      counter->freeRef();
    counter = other.counter;
    return *this;
  }


  inline T *operator -> ()
  { 
    if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return (T *)counter->ptr;
  }


  inline const T *operator -> () const 
  { 
    if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return (T *)counter->ptr;
  }


  inline bool operator ==(const TWrapper &p2) const
  { return  (   !counter && !p2.counter)
             || (counter && p2.counter && (counter->ptr==p2.counter->ptr));
  }
  

  inline bool operator ==(const TWrapped *p2) const
  { return    (!counter && !p2) 
           || ( counter && (counter->ptr==p2));
  }


  inline bool operator !=(const TWrapper &p2) const
  { return    (!counter &&  p2.counter)
           || ( counter && !p2.counter)
           || ( counter &&  p2.counter && (counter->ptr!=p2.counter->ptr));
  }


  inline bool operator !=(const TWrapped *p2) const
  { return    (!counter && p2)
           || ( counter && (counter->ptr!=p2));
  }

  inline bool operator < (const GCPtr<T> &ps) const
  { return    (!counter && ps.counter)
           || (int(counter->ptr) < int(ps.counter->ptr)); }


  inline T *getUnwrappedPtr()
  { return counter ? (T *)counter->ptr : NULL; }


  inline T &getReference()
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return (T &)*counter->ptr;
  }


  inline T const *getUnwrappedPtr() const
  { return counter ? (T const *)counter->ptr : NULL; }


  inline T &getReference() const
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return (T &)*counter->ptr;
  }


  template<class U>
  inline bool castable_to(const U *) const
  { return (dynamic_cast<U *>(counter->ptr)!=NULL); }


  template<class U>
  inline U *as(U *)
  { return counter ? dynamic_cast<U *>(counter->ptr) : NULL; }


  template<class U>
  inline const U *as(U *) const
  { return counter ? dynamic_cast<const U *>(counter->ptr) : NULL;  }
};


template<class T, class U>
inline bool castable_to(GCPtr<T> obj, U *)
{ return (dynamic_cast<U *>(obj.counter->ptr) != NULL); }


#define is_derived_from(x) castable_to((x *)NULL)
#define AS(x) as((x *)NULL)

#define CAST(o,x) ((o).counter ? dynamic_cast<x *>((o).counter->ptr) : NULL)

#define WRAPPEDVECTOR(x) GCPtrNML<vector<x> >


#endif
