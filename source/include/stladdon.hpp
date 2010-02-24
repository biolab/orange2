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


#ifndef __STLADDON_HPP
#define __STLADDON_HPP

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244 4996 4251)
  #if _MSC_VER < 1300
    #define _MSC_VER_60
  #else
    #define _MSC_VER_70
  #endif
#endif

#include <algorithm>
#include <vector>
#include <map>
#include <functional>
#include <string>
#include <limits>

using namespace std;

inline float sqr(const float &f1) { return f1*f1; }

#ifdef __GNUC__
#define stricmp strcasecmp
#endif

typedef vector<string> TIdList;

#define ILLEGAL_INT numeric_limits<int>::min()

/* Don't use quite_Nan (fell for that trick twice already!): quite_Nan equals anything */
#define ILLEGAL_FLOAT -1e30f

#define ITERATE(tpe, iter, pna)  for(tpe::iterator iter((pna).begin()), iter##_end((pna).end()); \
                                     iter!=iter##_end; iter++)

#define RITERATE(tpe, iter, pna)  for(tpe::reverse_iterator iter((pna).rbegin()), iter##_end((pna).rend()); \
                                     iter!=iter##_end; iter++)

#define PITERATE(tpe, iter, pna)  for(tpe::iterator iter((pna)->begin()), iter##_end((pna)->end()); \
                                     iter!=iter##_end; iter++)

#define const_ITERATE(tpe, iter, pna)  for(tpe::const_iterator iter((pna).begin()), iter##_end((pna).end()); \
                                           iter!=iter##_end; iter++)

#define const_PITERATE(tpe, iter, pna)  for(tpe::const_iterator iter((pna)->begin()), iter##_end((pna)->end()); \
                                     iter!=iter##_end; iter++)

#define this_ITERATE(iter)  for(iterator iter=begin(), iter##_end(end()); iter!=iter##_end; iter++)

#define const_this_ITERATE(iter)  for(const_iterator iter=begin(), iter##_end(end()); iter!=iter##_end; iter++)

template <class InputIterator, class T>
bool exists(InputIterator first, const InputIterator last, const T& value) 
{
  while (first != last && *first != value) ++first;
  return first!=last;
}

template <class T>
class TGenInt {
public:
  T n;
  TGenInt(const T &an=0): n(an) {};
  virtual int operator()() { return n++; }
};

template <class T, class U>
bool operator <(const pair<T, U> &p1, const pair<T, U> &p2)
{ return    (p1.first <p2.first)
         || (p1.first==p2.first) && (p1.second<p2.second);
}

template <class T, class U>
bool operator <(const pair<T, U> p1, const pair<T, U> p2)
{ return    (p1.first <p2.first)
         || (p1.first==p2.first) && (p1.second<p2.second);
}


template <class T, class Pred>
class predOn1st  {
public:
  bool operator()(const T p1, const T p2)
   { return Pred()(p1.first, p2.first); }
};


template <class T, class Pred>
class predOn2nd  {
public:
  bool operator()(const T p1, const T p2)
   { return Pred()(p1.second, p2.second); }
};


template <class T, class C>
bool exists(const T &cont, const C &value)   { return exists(cont.begin(), cont.end(), value); }


/* This is random_shuffle from GNU ISO C++,
   We needed to rewrite it because STL from Microsoft's C++ random_shuffle
   is unreadable and we want to make sure to have exactly the same function.
     
    Copyright (C) 2001, 2002 Free Software Foundation, Inc.
    Copyright (c) 1994 Hewlett-Packard Company
    Copyright (c) 1996 Silicon Graphics Computer Systems, Inc.
*/

template<typename RandomAccessIter, typename RandomNumberGenerator>
void or_random_shuffle(RandomAccessIter first, RandomAccessIter last, RandomNumberGenerator& rand)
{
  if (first == last)
    return;
  
  for (RandomAccessIter i = first + 1; i != last; ++i)
    iter_swap(i, first + rand((i - first)));
}

// Folowing two functions call sort and then randomly shuffle members with the same key value
//  The second functions uses predicates for testing order and equality (spr and epr)

/* Normaly, one wants sorting to be stable, i.e. that elements with equal keys
    remain in the same relative order as they were before. The random_sort function,
    to the contrary sort in such way that elements with equal keys are random_shuffled
    after sorting. The method should be used, for example, by attribute selection
    methods which sort attributes according to some measure to prevent that the attributes
    with the same value of criterion function are chosen according randomly.
*/
template<class RanIt>
void random_sort(RanIt first, RanIt last)
{ stable_sort(first, last);

  for(RanIt fs=first, ls; fs!=last; fs=ls) {
    for(ls=fs; ((++ls)!=last) && (*fs==*ls););
    or_random_shuffle(fs, ls);
  }
}

// Function that sorts and shuffles elements using common '<' and '==' relations; random function is given
template<class RanIt, class RandFunc>
void random_sort(RanIt first, RanIt last, RandFunc rf)
{ stable_sort(first, last);

  for(RanIt fs=first, ls; fs!=last; fs=ls) {
    for(ls=fs; ((++ls)!=last) && (*fs==*ls););
    or_random_shuffle(fs, ls, rf);
  }
}

/* Function that sorts and shuffles elements using given predicates for testing
    order (SPred) and equality (EPred) */
template<class RanIt, class SPred, class EPred>
void random_sort(RanIt first, RanIt last, SPred spr, EPred epr)
{ stable_sort(first, last, spr);

  for(RanIt fs=first, ls; fs!=last; fs=ls) {
    for(ls=fs; ((++ls)!=last) && epr(*fs,*ls););
    or_random_shuffle(fs, ls);
  }
}

/* Function that sorts and shuffles elements using given predicates for testing
    order (SPred) and equality (EPred) and a random function */
template<class RanIt, class SPred, class EPred, class RandFunc>
void random_sort(RanIt first, RanIt last, SPred spr, EPred epr, RandFunc rf)
{ stable_sort(first, last, spr);

  for(RanIt fs=first, ls; fs!=last; fs=ls) {
    for(ls=fs; ((++ls)!=last) && epr(*fs,*ls););
    or_random_shuffle(fs, ls, rf);
  }
}


#define VECTOR_INTERFACE_WOUT_OP(type, field) \
vector<type> field; \
typedef vector<type>::iterator iterator; \
typedef vector<type>::const_iterator const_iterator; \
typedef vector<type>::reverse_iterator reverse_iterator; \
typedef vector<type>::const_reverse_iterator const_reverse_iterator; \
typedef vector<type>::reference reference; \
typedef vector<type>::const_reference const_reference; \
typedef vector<type>::size_type size_type; \
typedef vector<type>::value_type value_type; \
\
reference       at(size_type i)  { return (field).at(i); } \
const_reference at(size_type i) const { return (field).at(i); } \
reference       back() { return (field).back(); } \
const_reference back() const { return (field).back(); } \
iterator        begin() { return (field).begin(); } \
const_iterator  begin() const { return (field).begin(); } \
void            clear() { (field).clear(); } \
bool            empty() const { return (field).empty(); } \
iterator        end() { return (field).end(); } \
const_iterator  end() const { return (field).end(); } \
iterator        erase(iterator it) { return (field).erase(it); } \
iterator        erase(iterator f, iterator l) { return (field).erase(f, l); } \
reference       front() { return (field).front(); } \
const_reference front() const { return (field).front(); } \
void            insert(iterator i_P, const type & x = type()) { (field).insert(i_P, x); } \
void            insert(iterator i_P, const_iterator i_F, const_iterator i_L) { (field).insert(i_P, i_F, i_L); } \
void            push_back(type const &x) { (field).push_back(x); } \
reverse_iterator rbegin() { return (field).rbegin(); } \
const_reverse_iterator rbegin() const { return (field).rbegin(); } \
reverse_iterator rend() { return (field).rend(); } \
const_reverse_iterator rend() const { return (field).rend(); } \
void            reserve(size_type n) { (field).reserve(n); } \
void            resize(size_type n, type x = type()) { (field).resize(n, x); } \
size_type       size() const { return (field).size(); } \
                operator const vector<type> &() const \
                { return field; }


#define VECTOR_INTERFACE(type, field) \
VECTOR_INTERFACE_WOUT_OP(type, field) \
reference       operator[](std::vector<type>::size_type i)  { return (field).operator[](i); } \
const_reference operator[](std::vector<type>::size_type i) const { return (field).operator[](i); }



#define PVECTOR_INTERFACE_WOUT_OP(type, field) \
WRAPPEDVECTOR(type) field; \
typedef std::vector<type>::iterator iterator; \
typedef std::vector<type>::const_iterator const_iterator; \
typedef std::vector<type>::reference reference; \
typedef std::vector<type>::reverse_iterator reverse_iterator; \
typedef std::vector<type>::const_reverse_iterator const_reverse_iterator; \
typedef std::vector<type>::const_reference const_reference; \
typedef std::vector<type>::size_type size_type; \
typedef std::vector<type>::value_type value_type; \
\
reference       at(vector<type>::size_type i)  { return (field)->at(i); } \
const_reference at(vector<type>::size_type i) const { return (field)->at(i); } \
reference       back() { return (field)->back(); } \
const_reference back() const { return (field)->back(); } \
iterator        begin() { return (field)->begin(); } \
const_iterator  begin() const { return (field)->begin(); } \
void            clear() { (field)->clear(); } \
bool            empty() const { return (field)->empty(); } \
iterator        end() { return (field)->end(); } \
const_iterator  end() const { return (field)->end(); } \
iterator        erase(iterator it) { return (field)->erase(it); } \
iterator        erase(iterator f, iterator l) { return (field)->erase(f, l); } \
reference       front() { return (field)->front(); } \
const_reference front() const { return (field)->front(); } \
void            insert(iterator i_P, const type & x = type()) { (field)->insert(i_P, x); } \
void            insert(iterator i_P, const_iterator i_F, const_iterator i_L) { (field)->insert(i_P, i_F, i_L); } \
void            push_back(const type &x) { (field)->push_back(x); } \
reverse_iterator rbegin() { return (field).rbegin(); } \
const_reverse_iterator rbegin() const { return (field).rbegin(); } \
reverse_iterator rend() { return (field).rend(); } \
const_reverse_iterator rend() const { return (field).rend(); } \
void            resize(size_type n, type x = type()) { (field).resize(n, x); } \
void            reserve(size_type n) { (field)->reserve(n); } \
size_type       size() const { return (field)->size(); } 


#define PVECTOR_INTERFACE(type, field) \
PVECTOR_INTERFACE_WOUT_OP(type, field) \
reference       operator[](vector<type>::size_type i)  { return (field)->operator[](i); } \
const_reference operator[](vector<type>::size_type i) const { return (field)->operator[](i); } 


/* tpdef below should be either "typedef typename" if the macro is used inside a template
   or only "typedef" if it's not */
#define MAP_INTERFACE_TYPES(type1,type2,field,tpdef) \
map<type1, type2> field; \
tpdef map<type1, type2>::iterator iterator; \
tpdef map<type1, type2>::const_iterator const_iterator; \
typedef pair<const type1, type2> value_type; \
tpdef map<type1, type2>::size_type size_type; \
typedef pair<iterator, bool> _Pairib;


#define MAP_INTERFACE_WOUT_OP(type1, type2, field, tpname) \
MAP_INTERFACE_TYPES(type1,type2,field, tpname) \
\
iterator        begin() { return (field).begin(); } \
const_iterator  begin() const { return (field).begin(); } \
iterator        end() { return (field).end(); } \
const_iterator  end() const { return (field).end(); } \
size_type       size() const {return (field).size(); } \
bool            empty() const {return (field).empty(); } \
void            clear() { (field).clear(); } \
size_type       count(const type1& i_Kv) const { return (field).count(i_Kv); } \
void            erase(iterator i_P) { (field).erase(i_P); } \
void            erase(iterator i_F, iterator i_L) { (field).erase(i_F, i_L); } \
iterator        find(const type1& i_Kv) { return (field).find(i_Kv); } \
const_iterator  find(const type1& i_Kv) const { return (field).find(i_Kv); } \
/*void            insert(iterator i_F, iterator i_L) { (field).insert(i_F, i_L); } */\
_Pairib         insert(const value_type& i_X) { return (field).insert(i_X); } \
iterator        insert(iterator i_P, const value_type& i_X) { return (field).insert(i_P, i_X); } \
iterator        lower_bound(const type1& i_Kv) { return (field).lower_bound(i_Kv); } \
const_iterator  lower_bound(const type1& i_Kv) const { return (field).lower_bound(i_Kv); } \
iterator        upper_bound(const type1& i_Kv) { return (field).upper_bound(i_Kv); } \
const_iterator  upper_bound(const type1& i_Kv) const { return (field).upper_bound(i_Kv); }

#define MAP_INTERFACE(type1, type2, field, tpname) \
MAP_INTERFACE_WOUT_OP(type1, type2, field, tpname) \
type2           operator[](const type1& i_Kv) { return (field)[i_Kv]; }

#endif
