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
     have traverse and dropReferences. (Doesn't need them and can't apply them
     since elements are not wrapped.)

     #define TIntList _TOrangeVector<int>
     VWRAPPER(IntList)


In both cases, you should declare the class's st_classDescription somewhere in
your code. See the handy macros below.

How to make a class vector-like?

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

#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267)
#endif

#include "garbage.hpp"
#include <vector>
#include "root.hpp"
#include "stladdon.hpp"

#define DEFINE_TOrangeVector_classDescription(_TYPE, _NAME) \
  TClassDescription TOrangeVector< _TYPE >::st_classDescription = { _NAME, &typeid(TOrangeVector< _TYPE >), &TOrange::st_classDescription, TOrange_properties, TOrange_components };

#define DEFINE__TOrangeVector_classDescription(_TYPE, _NAME) \
  TClassDescription _TOrangeVector< _TYPE >::st_classDescription = { _NAME, &typeid(_TOrangeVector< _TYPE >), &TOrange::st_classDescription, TOrange_properties, TOrange_components };


template<class T>
class _TOrangeVector : public TOrange
{ public:

    typedef typename vector<T>::iterator iterator;
    typedef typename vector<T>::const_iterator const_iterator;
    typedef typename vector<T>::reverse_iterator reverse_iterator;
    typedef typename vector<T>::const_reverse_iterator const_reverse_iterator;
    typedef typename vector<T>::reference reference;
    typedef typename vector<T>::const_reference const_reference;
    typedef typename vector<T>::size_type size_type;
    typedef typename vector<T>::value_type value_type;

    vector<T> __orvector;

    reference       at(size_type i)  { return (__orvector).at(i); }
    const_reference at(size_type i) const { return (__orvector).at(i); }
    reference       back() { return (__orvector).back(); }
    const_reference back() const { return (__orvector).back(); }
    iterator        begin() { return (__orvector).begin(); }
    const_iterator  begin() const { return (__orvector).begin(); }
    void            clear() { (__orvector).clear(); }
    bool            empty() const { return (__orvector).empty(); }
    iterator        end() { return (__orvector).end(); }
    const_iterator  end() const { return (__orvector).end(); }
    iterator        erase(iterator it) { return (__orvector).erase(it); }
    iterator        erase(iterator f, iterator l) { return (__orvector).erase(f, l); }
    reference       front() { return (__orvector).front(); }
    const_reference front() const { return (__orvector).front(); }

/*    iterator        insert(iterator i_P, const T & x = T()) { return (__orvector).insert(i_P, x); }
    void            insert(iterator i_P, const_iterator i_F, const_iterator i_L) { (__orvector).insert(i_P, i_F, i_L); }
*/
    void            insert(iterator i_P, const T & x = T()) { (__orvector).insert(i_P, x); }
    void            insert(iterator i_P, const_iterator i_F, const_iterator i_L) { (__orvector).insert(i_P, i_F, i_L); }

    void            push_back(T const &x) { (__orvector).push_back(x); }
    reverse_iterator rbegin() { return (__orvector).rbegin(); }
    const_reverse_iterator rbegin() const { return (__orvector).rbegin(); }
    reverse_iterator rend() { return (__orvector).rend(); }
    const_reverse_iterator rend() const { return (__orvector).rend(); }
    void            reserve(size_type n) { (__orvector).reserve(n); }
    void            resize(size_type n, T x = T()) { (__orvector).resize(n, x); }
    size_type       size() const { return (__orvector).size(); }
                    operator const vector<T> &() const { return __orvector; }
    reference       operator[](size_type i)  { return (__orvector).operator[](i); }
    const_reference operator[](size_type i) const { return (__orvector).operator[](i); }

    _TOrangeVector()
      {}

    _TOrangeVector(const size_type &i_N, const T &i_V = T())
      : __orvector(i_N, i_V)
      {}

    _TOrangeVector(const vector<T> &i_X)
      : __orvector(i_X)
      {}

    ~_TOrangeVector()
      {}

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }

    virtual TOrange *clone() const
      { return mlnew _TOrangeVector<T>(*this); }
};

#define TBoolList _TOrangeVector<bool>
#define TIntList _TOrangeVector<int>
#define TLongList _TOrangeVector<long>
#define TFloatList _TOrangeVector<float>
#define TIntFloatList _TOrangeVector<pair<int, float> >
#define TFloatFloatList _TOrangeVector<pair<float, float> >
#define TDoubleList _TOrangeVector<double>
#define TStringList _TOrangeVector<string>

#define VWRAPPER(x) typedef GCPtr< T##x > P##x;

VWRAPPER(BoolList)
VWRAPPER(IntList)
VWRAPPER(LongList)
VWRAPPER(FloatList)
VWRAPPER(IntFloatList)
VWRAPPER(FloatFloatList)
VWRAPPER(DoubleList)
VWRAPPER(StringList)

WRAPPER(Variable)

#include "values.hpp"

template<>
class _TOrangeVector<TValue> : public TOrange
{public:
  VECTOR_INTERFACE(TValue, __orvector);

  PVariable variable;

  _TOrangeVector(PVariable var = PVariable())
    : variable(var)
    {}
 
  _TOrangeVector(const vector<TValue>::size_type &i_N, const TValue &i_V = TValue(), PVariable var = PVariable())
    : __orvector(i_N, i_V), variable(var)
    {}

  _TOrangeVector(const vector<TValue> &i_X, PVariable var = PVariable())
    : __orvector(i_X), variable(var)
    {}

  _TOrangeVector(const _TOrangeVector<TValue> &other)
    : __orvector(other.__orvector), variable(other.variable)
    {}

  static TClassDescription st_classDescription;
  virtual TClassDescription const *classDescription() const;
  virtual TOrange *clone() const;

  int traverse(visitproc visit, void *arg) const
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

    typedef typename vector<T>::iterator iterator;
    typedef typename vector<T>::const_iterator const_iterator;
    typedef typename vector<T>::reverse_iterator reverse_iterator;
    typedef typename vector<T>::const_reverse_iterator const_reverse_iterator;
    typedef typename vector<T>::reference reference;
    typedef typename vector<T>::const_reference const_reference;
    typedef typename vector<T>::size_type size_type;
    typedef typename vector<T>::value_type value_type;

    vector<T> __orvector;

    reference       at(size_type i)  { return (__orvector).at(i); }
    const_reference at(size_type i) const { return (__orvector).at(i); }
    reference       back() { return (__orvector).back(); }
    const_reference back() const { return (__orvector).back(); }
    iterator        begin() { return (__orvector).begin(); }
    const_iterator  begin() const { return (__orvector).begin(); }
    void            clear() { (__orvector).clear(); }
    bool            empty() const { return (__orvector).empty(); }
    iterator        end() { return (__orvector).end(); }
    const_iterator  end() const { return (__orvector).end(); }
    iterator        erase(iterator it) { return (__orvector).erase(it); }
    iterator        erase(iterator f, iterator l) { return (__orvector).erase(f, l); }
    reference       front() { return (__orvector).front(); }
    const_reference front() const { return (__orvector).front(); }
    void            insert(iterator i_P, const T &x = T()) { (__orvector).insert(i_P, x); }
    void            insert(iterator i_P, const_iterator i_F, const_iterator i_L) { (__orvector).insert(i_P, i_F, i_L); }
    void            push_back(T const &x) { (__orvector).push_back(x); }
    reverse_iterator rbegin() { return (__orvector).rbegin(); }
    const_reverse_iterator rbegin() const { return (__orvector).rbegin(); }
    reverse_iterator rend() { return (__orvector).rend(); }
    const_reverse_iterator rend() const { return (__orvector).rend(); }
    void            reserve(size_type n) { (__orvector).reserve(n); }
    void            resize(size_type n, T x = T()) { (__orvector).resize(n, x); }
    size_type       size() const { return (__orvector).size(); }
                    operator const vector<T> &() const { return __orvector; }
    reference       operator[](size_type i)  { return (__orvector).operator[](i); }
    const_reference operator[](size_type i) const { return (__orvector).operator[](i); }


    TOrangeVector()
      {}
 
    TOrangeVector(const size_type &i_N, const T &i_V = T())
      : __orvector(i_N, i_V)
      {}

    TOrangeVector(const vector<T> &i_X)
      : __orvector(i_X)
      {}

    TOrangeVector(const TOrangeVector<T> &other)
      : TOrange(other),
        __orvector(other.__orvector)
      {}

    int traverse(visitproc visit, void *arg) const
    { TRAVERSE(TOrange::traverse);
      for(const_iterator be=begin(), ee=end(); be!=ee; be++)
        PVISIT(*be);
      return 0;
    }

    int dropReferences()
    { DROPREFERENCES(TOrange::dropReferences);
      clear();
      return 0;
    }

    static TClassDescription st_classDescription;

    virtual TClassDescription const *classDescription() const
      { return &st_classDescription; }

    virtual TOrange *clone() const
      { return mlnew TOrangeVector<T>(*this); }
};


#define TVarList TOrangeVector<PVariable> 
VWRAPPER(VarList)

class TAttributedFloatList : public _TOrangeVector<float>
{
public:
  PVarList attributes;

  TAttributedFloatList()
  {}

  TAttributedFloatList(PVarList vlist)
  : attributes(vlist)
  {}

  TAttributedFloatList(PVarList vlist, const size_type &i_N, const float &f = 0.0)
  : _TOrangeVector<float>(i_N, f),
    attributes(vlist)
  {}

  TAttributedFloatList(PVarList vlist, const vector<float> &i_X)
  : _TOrangeVector<float>(i_X),
    attributes(vlist)
  {}

  static TClassDescription st_classDescription;
  virtual TClassDescription const *classDescription() const;
  virtual TOrange *clone() const;
};


class TAttributedBoolList : public _TOrangeVector<bool>
{
public:
  PVarList attributes;

  TAttributedBoolList()
  {}

  TAttributedBoolList(PVarList vlist)
  : attributes(vlist)
  {}

  TAttributedBoolList(PVarList vlist, const size_type &i_N, const bool b= false)
  : _TOrangeVector<bool>(i_N, b),
    attributes(vlist)
  {}

  TAttributedBoolList(PVarList vlist, const vector<bool> &i_X)
  : _TOrangeVector<bool>(i_X),
    attributes(vlist)
  {}

  static TClassDescription st_classDescription;
  virtual TClassDescription const *classDescription() const;
  virtual TOrange *clone() const;
};


/* This is to fool pyprops.py
#define TAttributedFloatList _TOrangeVector<float>
#define TAttributedBoolList _TOrangeVector<bool>
*/
VWRAPPER(AttributedFloatList)
VWRAPPER(AttributedBoolList)

#endif
