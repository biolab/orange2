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
#include "errors.hpp"

#ifdef _MSC_VER
  #include <crtdbg.h>
  #pragma warning (disable : 4786 4114 4018 4267)
  #pragma warning (disable : 4127) // conditional expression is constant (reported by _ASSERT without debug)
  #define TYPENAME(x) (x).name()+7

#else // !_MSC_VER
  #include <assert.h>
  #define _ASSERT assert
  char *demangle(const type_info &type);
  #define TYPENAME(x) demangle(x)+1
#endif

#define mlnew new
#define mldelete delete

class TOrangeType;
extern TOrangeType PyOrOrange_Type;

extern PyTypeObject PyOrNonOrange_Type;

template<class T>
class TGCCounter {
public:
	PyObject_HEAD
  T *ptr;
  PyObject *orange_dict;
  bool call_constructed, is_reference;

  inline void getRef()
  { Py_INCREF(this); }

  inline void freeRef()
  { Py_DECREF(this); }
};


template<class T>
class TGCCounterNML {
public:
	PyObject_HEAD
  T *ptr;
  PyObject *notorange_dict;
  bool is_reference;

  typedef void (*TDestructor)(void *);
  TDestructor destructor;

  inline void getRef()
  { Py_INCREF(this); }

  inline void freeRef()
  { Py_DECREF(this); }
};



/* Checks whether the object is gc-tracked or not.
   This way, we distinguish between wrapped pointers (which are)
   and wrapped references (which are not wrapped).
   The latter don't get freed. */

#define PyObject_IsPointer(o) (((PyGC_Head *)(o)-1)->gc.gc_next != NULL)
#define PyObject_IsReference(o) (((PyGC_Head *)(o)-1)->gc.gc_next == NULL)
  


class TOrange;
typedef TGCCounter<TOrange> TPyOrange;

#define VISIT(obj) { int res=visit((PyObject *)(obj), arg); if (res) return res; }

template<class T>
class GCPtr {
public:
  typedef TGCCounter<T> GCCounter;

  GCCounter *counter;


  GCPtr() 
  : counter(NULL)
  {};


  void init() // used when GCPtr is allocated in C code and constructor is not called
  { counter = NULL; }


  GCPtr(GCCounter *acounter, bool)  // Used by PyOrange_AS_Orange
  : counter(acounter)
  { if (counter)
      counter->getRef(); 
  }


  template<class U>
  GCPtr(U *ptr)
  : counter(NULL)
  { if (ptr)
      if (ptr->myWrapper)
        counter = reinterpret_cast<GCCounter *>(ptr->myWrapper);
      else {
        counter = PyObject_GC_New(GCCounter, (PyTypeObject *)&PyOrOrange_Type);
        counter->orange_dict = NULL;
        counter->call_constructed = false;
        counter->is_reference = false;
        counter->ptr = ptr;
        ptr->myWrapper = reinterpret_cast<TGCCounter<TOrange> *>(counter);
        PyObject_GC_Track(counter);
      }
  }


  template<class U>
  GCPtr(U *ptr, PyTypeObject *type)
  : counter(NULL)
  { if (ptr) {
      counter = (GCCounter *)type->tp_alloc(type, 0);
      counter->orange_dict = NULL;
      counter->call_constructed = false;
      counter->is_reference = false;
      counter->ptr = ptr;
      _ASSERT(!ptr->myWrapper);
      ptr->myWrapper = reinterpret_cast<TGCCounter<TOrange> *>(counter);
    }
  }


  explicit GCPtr(T &ptr) 
  : counter(NULL)
  { counter = PyObject_GC_New(GCCounter, (PyTypeObject *)&PyOrOrange_Type);
    counter->orange_dict = NULL;
    counter->call_constructed = false;
    counter->is_reference = true; // this should never be deleted
    counter->ptr = &ptr;
    /* No calling PyObject_GC_Track(counter);
       These object cannot be freed; if they participate in a cycle, the cycle
       will disappear as soon as they die (and these objects die soon).
       We even distinguish between wrapped pointers and reference by checking
       whether they are gc-tracked or not. */
  }


  template<class U>
  GCPtr(const GCPtr<U> &other)
  : counter((GCCounter *)(other.counter)) 
  { if (counter) {
      if (!dynamic_cast<T *>(other.counter->ptr))
        raiseError("bad cast from %s to %s", typeid(U).name(), typeid(T).name());
      counter->getRef(); 
    }
  }


  GCPtr(const GCPtr<T> &other)
  : counter(other.counter)
  { if (counter)
      counter->getRef();
  }


  GCPtr<T> & operator =(const GCPtr<T> &other)
  { if (other.counter)
      other.counter->getRef();
    if (counter)
      counter->freeRef();
    counter = other.counter;
    return *this;
  }


  ~GCPtr()
  { if (counter)
      counter->freeRef();
  }


  inline void mark(const int &n=1) const
  {}


  T *operator -> ()
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return counter->ptr;
  }


  const T *operator -> () const 
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return counter->ptr;
  }


  bool operator ==(const GCPtr<T> &p2) const
  { return  (   !counter && !p2.counter)
             || (counter && p2.counter && (counter->ptr==p2.counter->ptr));
  }
  

  bool operator ==(const T *p2) const
  { return    (!counter && !p2) 
           || ( counter && (counter->ptr==p2));
  }


  bool operator !=(const GCPtr<T> &p2) const
  { return    (!counter &&  p2.counter)
           || ( counter && !p2.counter)
           || ( counter &&  p2.counter && (counter->ptr!=p2.counter->ptr));
  }


  bool operator !=(const T *p2) const
  { return    (!counter && p2)
           || ( counter && (counter->ptr!=p2));
  }

  bool operator < (const GCPtr<T> &ps) const
  { return    (!counter && ps.counter)
           || (int(counter->ptr) < int(ps.counter->ptr)); }


  operator bool() const
  { return (counter!=NULL); }


  inline T const *getUnwrappedPtr() const
  { return counter ? counter->ptr : NULL; }


  inline T &getReference() const
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return *counter->ptr;
  }


  template<class U>
  bool dynamic_cast_to(U *&ptr)
  { return ( (ptr = counter ? dynamic_cast<U *>(counter->ptr) : NULL) != NULL); }


  template<class U>
  bool dynamic_cast_to(U *&ptr) const
  { return ( (ptr = counter ? dynamic_cast<U *>(counter->ptr) : NULL) != NULL); }


  template<class U>
  inline bool castable_to(const U *) const
  { return (dynamic_cast<U *>(counter->ptr)!=NULL); }


  template<class U>
  inline bool is(const U *) const
  { return (typeid(U)==typeid(T)) != 0; }


  template<class U>
  U *as(U *)
  { return counter ? dynamic_cast<U *>(counter->ptr) : NULL; }


  template<class U>
  const U *as(U *) const
  { return counter ? dynamic_cast<const U *>(counter->ptr) : NULL;  }
};



extern PyTypeObject PyNotOrOrange_Type;

template<class T>
class GCPtrNML {
public:
  typedef TGCCounterNML<T> GCCounter;

  static typename GCCounter::TDestructor destructor;


  GCCounter *counter;


  GCPtrNML() 
  : counter(NULL)
  {}


  void init() // used when GCPtr is allocated in C code and constructor is not called
  { counter = NULL; }

  GCPtrNML(GCCounter *acounter, bool) 
  : counter(acounter)
  { if (counter)
      counter->getRef(); }


  template<class U>
  GCPtrNML(U *ptr)
  : counter(NULL)
  { if (ptr) {
      counter = PyObject_New(GCCounter, (PyTypeObject *)&PyNotOrOrange_Type);
      counter->notorange_dict=NULL;
      counter->is_reference = false;
      counter->destructor=destructor;
      counter->ptr=ptr;
    }
  }


  GCPtrNML(T &ptr) 
  : counter(NULL)
  { counter = PyObject_New(GCCounter, (PyTypeObject *)&PyNotOrOrange_Type);
    counter->notorange_dict = NULL;
    counter->destructor = destructor;
    counter->is_reference = true;
    counter->ptr = &ptr;
  }


  template<class U>
  GCPtrNML(const GCPtrNML<U> &other)
  : counter((GCCounter *)(other.counter)) 
  { if (counter)
      if (!dynamic_cast<T *>(other.counter->ptr))
        raiseError("bad cast from %s to %s", typeid(U).name(), typeid(T).name());
      else
        counter->getRef(); 
  }


  GCPtrNML(const GCPtrNML<T> &other)
  : counter(other.counter)
  { if (counter)
      counter->getRef();
  }


  GCPtrNML<T> & operator =(const GCPtrNML<T> &other)
  { if (other.counter)
      other.counter->getRef();
    if (counter)
      counter->freeRef();
    counter=other.counter;
    return *this;
  }


  ~GCPtrNML()
  { if (counter)
      counter->freeRef();
  }


  inline void mark(const int &n=1) const
  {}


  inline void unbind()
  {}


  T *operator -> ()
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return counter->ptr;
  }


  const T *operator -> () const 
  { if (!counter)
      raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return counter->ptr;
  }


  bool operator ==(const GCPtr<T> &p2) const
  { return    (!counter && !p2.counter)
           || (counter && p2.counter && (counter->ptr==p2.counter->ptr));
  }
  

  bool operator ==(const T *p2) const
  { return    (!counter && !p2) 
           || (counter && (counter->ptr==p2));
  }


  bool operator !=(const GCPtr<T> &p2) const
  { return    (!counter && p2.counter)
           || (counter && !p2.counter)
           || (counter->ptr!=p2.counter->ptr);
  }


  bool operator !=(const T *p2) const
  { return counter ? (counter->ptr!=p2) : !p2; }


  operator bool() const
  { return (counter!=NULL); }


  inline T const *getUnwrappedPtr() const
  { return counter->ptr; }


  inline T &getReference() const
  { if (!counter)
       raiseError("Orange internal error: NULL pointer to '%s'", TYPENAME(typeid(T)));
    return *counter->ptr;
  }


  template<class U>
  bool dynamic_cast_to(U *&ptr)
  { return ( (ptr = counter ? dynamic_cast<U *>(counter->ptr) : NULL) != NULL); }


  template<class U>
  bool dynamic_cast_to(U *&ptr) const
  { return ( (ptr = counter ? dynamic_cast<U *>(counter->ptr) : NULL) != NULL); }


  template<class U>
  inline bool castable_to(const U *) const
  { return (dynamic_cast<U *>(counter->ptr)!=NULL); }


  template<class U>
  inline bool is(const U *) const
  { return (typeid(U)==typeid(T))!=0; }


  template<class U>
  U *as(U *)
  { return counter ? dynamic_cast<U *>(counter->ptr) : NULL; }


  template<class U>
  const U *as(U *) const
  { return counter ? dynamic_cast<const U *>(counter->ptr) : NULL; }
};



#define DEFINE_DESTRUCTOR(type) \
void type##_destructor(void *x) { mldelete (type *)x; } \
TGCCounterNML<type>::TDestructor GCPtrNML<type>::destructor = type##_destructor;


#define is_derived_from(x) castable_to((x *)NULL)
#define IS(x) is((x *)NULL)
#define AS(x) as((x *)NULL)

#define WRAPPER(x) class T##x; typedef GCPtr<T##x> P##x;

#define WRAPPEDNML(x) GCPtrNML<T##x>
#define WRAPPERNML(x) typedef WRAPPEDNML(x) P##x;
#define WRAPPEDVECTOR(x) GCPtrNML<vector<x> >


// This is defined by Python but then redefined by STLPort
#undef LONGLONG_MAX
#undef ULONGLONG_MAX
