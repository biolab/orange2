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

#include <vector>
#include "root.hpp"
#include "stladdon.hpp"

#ifdef _MSC_VER
  #define DEFINE_TOrangeVector_classDescription(_TYPE, _NAME, _WRAPPED, _API) \
    EXPIMP_TEMPLATE template class _API TOrangeVector< _TYPE, _WRAPPED >; \
    _API TClassDescription TOrangeVector< _TYPE, _WRAPPED >::st_classDescription = { _NAME, &typeid(TOrangeVector< _TYPE, _WRAPPED >), &TOrange::st_classDescription, TOrange_properties, TOrange_components }; 

  #ifndef __PLACEMENT_NEW_INLINE
    #define __PLACEMENT_NEW_INLINE
    inline void * operator new(size_t, void *_P)	{return (_P); }
  #endif

#else
  #define DEFINE_TOrangeVector_classDescription(_TYPE, _NAME, _WRAPPED, _API) \
    template<> \
    TClassDescription TOrangeVector< _TYPE, _WRAPPED >::st_classDescription = { _NAME, &typeid(TOrangeVector< _TYPE, _WRAPPED >), &TOrange::st_classDescription, TOrange_properties, TOrange_components };
#endif


int ORANGE_API _RoundUpSize(const int &n);

template<class T, bool Wrapped = true>
class TOrangeVector : public TOrange
{ public:
    typedef T *iterator;
    typedef T const *const_iterator;

    iterator _First, _Last, _End;

    static TClassDescription st_classDescription;
    virtual TClassDescription const *classDescription() const     { return &st_classDescription; }
    virtual TOrange *clone() const                                { return mlnew TOrangeVector<T, Wrapped>(*this); }

    class reverse_iterator {
    public:
      iterator position;

      explicit reverse_iterator(iterator p)  : position(p) {}
      reverse_iterator(const reverse_iterator &old) : position(old.position) {}

      reverse_iterator &operator ++()
      { --position;
        return *this; 
      }

      reverse_iterator operator ++(int)
      { reverse_iterator sv = *this;
        position--;
        return sv;
      }

      reverse_iterator &operator --()
      { ++position;
        return *this;
      }

      reverse_iterator operator --(int)
      { reverse_iterator sv = *this;
        position++;
        return sv;
      }

      T &operator *() const
      { return position[-1]; }

      T *operator->() const
      { return (&**this); }

      reverse_iterator operator +(const int &N)
      { return reverse_iterator(position - N); }

      reverse_iterator operator -(const int &N)
      { return reverse_iterator(position + N); }

      int operator -(const reverse_iterator &other) const
      { return other.position - position; }

      reverse_iterator &operator +=(const int &N)
      { position -= N;
        return *this;
      }

      reverse_iterator &operator -=(const int &N)
      { position += N;
        return *this;
      }

      bool operator == (const reverse_iterator &other) const
      { return position == other.position; }

      bool operator != (const reverse_iterator &other) const
      { return position != other.position; }

      bool operator < (const reverse_iterator &other) const
      { return position > other.position; }

      bool operator <= (const reverse_iterator &other) const
      { return position >= other.position; }

      bool operator > (const reverse_iterator &other) const
      { return position < other.position; }

      bool operator >= (const reverse_iterator &other) const
      { return position <= other.position; }
    };

    inline void _Set(const iterator &p, const T &X) const
    { new ((void *)p) T(X); }

    TOrangeVector<T, Wrapped>()
    : _First(NULL), _Last(NULL), _End(NULL)
    {}


    TOrangeVector<T, Wrapped>(const int &N, const T &V = T())
    : _First(NULL), _Last(NULL), _End(NULL)
    {
      _Resize(N);
      int n = N;
      for(; n--; _Set(_Last++, V));
    }


    TOrangeVector<T, Wrapped>(const TOrangeVector<T, Wrapped> &old)
    : _First(NULL), _Last(NULL), _End(NULL)
    {
      _Resize(old.size());
      for(const_iterator r = old._First; r != old._Last; _Set(_Last++, *(r++)));
    }
     
     
    TOrangeVector<T, Wrapped>(const vector<T> &old)
    : _First(NULL), _Last(NULL), _End(NULL)
    {
      _Resize(old.size());
      for(typename vector<T>::const_iterator vi(old.begin()), vi_end(old.end()); vi != vi_end; _Set(_Last++, *(vi++)));
    }


    TOrangeVector<T, Wrapped> &operator =(const TOrangeVector<T, Wrapped> old)
    { 
      _Destroy(_First, _Last);
      _Resize(old.size());
      for(iterator f = old._First; f != old._Last; _Set(_Last++, *(f++)));
      return *this;
    }


    ~TOrangeVector<T, Wrapped>()
    { 
      _Destroy(_First, _Last);
      free(_First);
      _First = _Last = _End = NULL;
    }


    operator vector<T>() const
    {
      vector<T> conv;
      conv.resize(size());
      int i = 0;
      for(iterator p = _First; p != _Last; conv[i++] = *(p++));
      return conv;
    }


    virtual int traverse(visitproc visit, void *arg) const
    { 
      TRAVERSE(TOrange::traverse);
      if (Wrapped)
        for(const_iterator be=begin(), ee=end(); be!=ee; be++)
          PVISIT(*(const GCPtr<TOrange> *)&*be);
      return 0;
    }

    virtual int dropReferences()
    { DROPREFERENCES(TOrange::dropReferences);
      clear();
      return 0;
    }

    iterator begin()                       { return _First; }
    const_iterator begin() const           { return _First; }
    reverse_iterator rbegin()              { return reverse_iterator(end()); }
    iterator end()                         { return _Last; }
    const_iterator end() const             { return _Last; }
    reverse_iterator rend()                { return reverse_iterator(begin()); }

    T &back()                              { return _Last[-1]; }
    const T &back() const                  { return _Last[-1]; }
    T &front()                             { return *_First; }
    const T &front() const                 { return *_First; }

    T &operator[](const int i)             { return _First[i]; }
    const T &operator[](const int i) const { return _First[i]; }
    
    bool empty() const                     { return _First == _Last; }
    int size() const                       { return _Last - _First; }

    T &at(const int &N)
    { if (N >= size())
        raiseError("vector subscript out of range");
      return _First[N];
    }

    const T &at(const int &N) const
    { if (N >= size())
        raiseError("vector subscript out of range");
      return _First[N];
    }

    void clear()
    { _Destroy(_First, _Last);
      _First = _End = _Last = NULL;
    }

    iterator erase(iterator it)
    { 
      it->~T();
      memmove(it, it+1, (_Last - it - 1) * sizeof(T));
      _Last--;
      return it;
    }

    iterator erase(iterator first, iterator last)
    { 
      if (first != last) {
        _Destroy(first, last);
        if (last != _Last)
          memmove(first, last, (_Last - last - 1) * sizeof(T));
        _Last -= last - first;
      }
      return first;
    }

    
    iterator insert(iterator p, const T &X = T())
    { 
      const int ind = p - _First;
      insert(p, 1, X);
      return _First + ind;        
    }


    void insert(iterator p, const int &n, const T &X)
    {
      if (_End - _Last < n)
        _Resize(size() + n);

      iterator e = p + n;
      if (p != _Last)
        memmove(e, p, (_Last - p - 1) * sizeof(T));

      for(; p != e; _Set(p++, X));
      _Last += n;
    }


    void insert(iterator p, iterator first, iterator last)
    {
      const int n = last - first;
      if (_End - _Last < n)
        _Resize(size() + n);
      
      iterator e =  p + n;
      if (p != _Last)
        memmove(e, p, (_Last - p - 1) * sizeof(T));

      for(; first != last; _Set(p++, *(first++)));
      _Last += n;
    }


    void push_back(T const &x)
    {  
       if (_Last == _End)
        _Resize(size() + 1);
      _Set(_Last++, x);
    }

    void reserve(const int n)
    { if (n >= _Last - _First)
        _Resize(n);
    }

    void resize(const int n, T x = T())
    { if (n < size()) {
        _Destroy(_First + n, _Last);
        _Resize(n);
        _Last = _First + n;
      }
      else {
        _Resize(n);
        for(iterator _nLast = _First + n; _Last != _nLast; _Set(_Last++, x));
      }
    }

    void _Destroy(const iterator first, const iterator last)
    { for(iterator p = first; p != last; p++)
        p->~T(); 
    }


    void _Resize(const int &n)
    {
      int sze = _RoundUpSize(n);
      if (!_First) {
        _Last = _First = (iterator)malloc(sze * sizeof(T));
        _End = _First + sze;
      }
      else if (_End - _First != sze) {
        int osize = size();
        _First = (iterator)realloc(_First, sze * sizeof(T));
        _Last = _First + osize;
        _End = _First + sze;
      }
    }
};


/*EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<bool, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<int, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<long, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<float, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<int, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<pair<int, float>, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<pair<float, float>, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<double, false>;
EXPIMP_TEMPLATE template class ORANGE_API TOrangeVector<string, false>;
*/
#define TBoolList TOrangeVector<bool, false>
#define TIntList TOrangeVector<int, false>
#define TLongList TOrangeVector<long, false>
#define TFloatList TOrangeVector<float, false>
#define TIntFloatList TOrangeVector<pair<int, float>, false >
#define TFloatFloatList TOrangeVector<pair<float, float>, false >
#define TDoubleList TOrangeVector<double, false>
#define TStringList TOrangeVector<string, false>

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
#include "vars.hpp"

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable: 4275)
#endif

/* This is to fool pyprops
#define TValueList _TOrangeVector<float>
*/

class ORANGE_API TValueList : public TOrangeVector<TValue, false>
{
public:
  __REGISTER_CLASS

  PVariable variable; //P the variable to which the list applies

  TValueList(PVariable var = PVariable())
  : TOrangeVector<TValue, false>(),
    variable(var)
  {}
 
  TValueList(const int &N, const TValue &V = TValue(), PVariable var = PVariable())
  : TOrangeVector<TValue, false>(N, V),
    variable(var)
  {}

  TValueList(const TOrangeVector<TValue, false> &i_X, PVariable var = PVariable())
  : TOrangeVector<TValue, false>(i_X),
    variable(var)
  {}

  TValueList(const TValueList &other)
  : TOrangeVector<TValue, false>(other),
    variable(other.variable)
  {}

  int traverse(visitproc visit, void *arg) const
  { 
    TRAVERSE(TOrange::traverse);

    for(TValue *p = _First; p != _Last; p++)
      if (p->svalV)
        PVISIT(p->svalV);

    PVISIT(variable);
    return 0;
  }

  int dropReferences()
  { DROPREFERENCES(TOrange::dropReferences);
    variable = PVariable();
    return 0;
  }
};


WRAPPER(ValueList)


#define TVarList TOrangeVector<PVariable> 
VWRAPPER(VarList)


class ORANGE_API TAttributedFloatList : public TOrangeVector<float, false>
{
public:
  __REGISTER_CLASS

  PVarList attributes;

  TAttributedFloatList()
  {}

  TAttributedFloatList(PVarList vlist)
  : attributes(vlist)
  {}

  TAttributedFloatList(PVarList vlist, const int &i_N, const float &f = 0.0)
  : TOrangeVector<float, false>(i_N, f),
    attributes(vlist)
  {}

  TAttributedFloatList(PVarList vlist, const vector<float> &i_X)
  : TOrangeVector<float,false>(i_X),
    attributes(vlist)
  {}
};


class ORANGE_API TAttributedBoolList : public TOrangeVector<bool, false>
{
public:
  __REGISTER_CLASS

  PVarList attributes;

  TAttributedBoolList()
  {}

  TAttributedBoolList(PVarList vlist)
  : attributes(vlist)
  {}

  TAttributedBoolList(PVarList vlist, const int &i_N, const bool b= false)
  : TOrangeVector<bool, false>(i_N, b),
    attributes(vlist)
  {}

  TAttributedBoolList(PVarList vlist, const vector<bool> &i_X)
  : TOrangeVector<bool, false>(i_X),
    attributes(vlist)
  {}
};

#ifdef _MSC_VER
  #pragma warning(pop)
#endif

/* This is to fool pyprops.py
#define TAttributedFloatList _TOrangeVector<float>
#define TAttributedBoolList _TOrangeVector<bool>
*/
WRAPPER(AttributedFloatList)
WRAPPER(AttributedBoolList)


template class ORANGE_API std::vector<int>;
template class ORANGE_API std::vector<float>;

/* These are defined as classes, not templates, so that 
class TIntIntPair {
public:
  int first, second;
  TIntIntPair(const int &f, const int &s)
  : first(f),
    second(s)
  {}
};

class TIntIntPair {
public:
  int first, second;
  TIntIntPair(const int &f, const int &s)
  : first(f),
    second(s)
  {}
};

class TIntIntPair {
public:
  int first, second;
  TIntIntPair(const int &f, const int &s)
  : first(f),
    second(s)
  {}
};

class TIntIntPair {
public:
  int first, second;
  TIntIntPair(const int &f, const int &s)
  : first(f),
    second(s)
  {}
};
*/

#ifdef _MSC_VER
  template class ORANGE_API std::vector<pair<int, int> >;
  template class ORANGE_API std::vector<int>;
#endif

#endif
