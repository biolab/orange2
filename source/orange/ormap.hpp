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

How to wrap a vector so that it can become a property?

  It depends upon the kind of elements the vector contains.

  1. Say that vector contains Orange objects, for example TVariables.
     You cannot (well, should not) have vectors of pointers to Orange
     objects, but always vectors of wrapped orange objects. You don't
     aggregate TVariable *, but always PVariable. (You could, however,
     aggregate TVariable, but I think I've never done so.)

     Instead of vector, you should use TOrangeVector. It is a class that
     inherits from TOrange which enables it to be used as property of
     other objects, but also behaves as a vector (it uses macro
     VECTOR_INTERFACE to define vector methods like push_back, clear, ...).
     Besides, it defines traverse and dropReferences.

     You should not use typedef to define a new type for you vector
     and WRAPPER to wrap it. Instead, you define it with

     #define TVarList TOrangeVector<PVariable>
     VWRAPPER(VarList)

  2. If vector contains non-Orange types, elements should not be wrapped.
     (In general, don't wrap non-Orange types. If you need to wrap them,
     color them orange.)

     Story is the same as above, except that you take _TOrangeVector instead
     of TOrangeVector. The only difference is that _TOrangeVector does not
     have traverse and dropReferences. (Doesn't need them and can apply them
     since elements are not wrapped.)

     #define TIntList _TOrangeVector<int>
     VWRAPPER(IntList)

How to make some class vector-like?

  Simply include VECTOR_INTERFACE(TElementType, field-name).
  Field-name is not used often since VECTOR_INTERFACE provides the class with
  vector-like methods that transparently access the field-name. You need to use
  it in constructor, however, if you want to initialize the vector to non-default.

  Warning: If you do this and if vector contains wrapped oranges,
  you should write traverse and dropReferences (you'll get a memory leak otherwise)

Vectors of non-wrapped elements (point 2) should be declared in this header.
Vectors of wrapped types should be declared in corresponding headers
(TVarList, for instance, is declared in vars.hpp).

For instructions on exporting those vectors to Python, see vectortemplates.hpp.
*/


#ifndef __ORVECTOR_HPP
#define __ORVECTOR_HPP

#include "garbage.hpp"
#include <vector>
#include "root.hpp"
#include "stladdon.hpp"

template<class T>
class _TOrangeMap : public TOrange
{ public:
    MAP_INTERFACE(T, __ormap);

    _TOrangeMap()
      {}
 
    _TOrangeVector(const vector<T>::size_type &_N, const T &_V = T())
      : __orvector(_N, _V)
      {}

    _TOrangeVector(const vector<T>::_Myt& _X)
      : __orvector(_X)
      {}
};

#define TBoolList _TOrangeVector<bool>
#define TIntList _TOrangeVector<int>
#define TLongList _TOrangeVector<long>
#define TFloatList _TOrangeVector<float>
#define TFloatFloatList _TOrangeVector<pair<float, float> >
#define TDoubleList _TOrangeVector<double>
#define TStringList _TOrangeVector<string>

#define VWRAPPER(x) typedef GCPtr< T##x > P##x;

VWRAPPER(BoolList)
VWRAPPER(IntList)
VWRAPPER(LongList)
VWRAPPER(FloatList)
VWRAPPER(FloatFloatList)
VWRAPPER(DoubleList)
VWRAPPER(StringList)

WRAPPER(Variable)

template<>
class _TOrangeVector<TValue> : public TOrange
{public:
  PVariable variable;

  VECTOR_INTERFACE(T, __orvector);

  _TOrangeVector(PVariable var = PVariable())
    : variable(var)
    {}
 
  _TOrangeVector(const vector<TValue>::size_type &_N, const TValue &_V = TValue(), PVariable var = PVariable())
    : __orvector(_N, _V), variable(var)
    {}

  _TOrangeVector(const vector<TValue>::_Myt& _X, PVariable var = PVariable())
    : __orvector(_X), variable(var)
    {}

  int traverse(visitproc visit, void *arg)
  { TRAVERSE(TOrange::traverse);
    PVISIT(variable);
    return 0;
  }

  int dropReferences()
  { DROPREFERENCES(TOrange::dropReferences);
    variable = PVariable();
    return 0;
  }
};

#define TValueList _TOrangeVector<TValue>
VWRAPPER(ValueList)



template<class T>
class TOrangeVector : public TOrange
{ public:
    VECTOR_INTERFACE_W_TEMPLATE(T, __orvector); 

    TOrangeVector()
      {}
 
    TOrangeVector(const vector<T>::size_type &_N, const T &_V = T())
      : __orvector(_N, _V)
      {}

    TOrangeVector(const vector<T>::_Myt& _X)
      : __orvector(_X)
      {}

    int traverse(visitproc visit, void *arg)
    { TRAVERSE(TOrange::traverse);
      for(iterator be=begin(), ee=end(); be!=ee; be++)
        PVISIT(*be);
      return 0;
    }

    int dropReferences()
    { DROPREFERENCES(TOrange::dropReferences);
      clear();
      return 0;
    }
};

#endif