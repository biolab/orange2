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


#ifndef __TABLE_HPP
#define __TABLE_HPP

#include "examplegen.hpp"
#include "random.hpp"

class TFilter;

// Example generator which stores examples in STL's vector.
class TExampleTable : public TExampleGenerator {
public:
  __REGISTER_CLASS

  TExample **examples;
  TExample **_Last, **_EndSpace;
  PRandomGenerator randomGenerator; //P random generator used by randomExample

  // Iterates through examples of basevector
  #define baseITERATE(x) ITERATE(vector<TExample>, x, examples)

  TExampleTable(PDomain);
  TExampleTable(PExampleGenerator orig);
  TExampleTable(PDomain, PExampleGenerator orig);
  ~TExampleTable();

protected:
  TExampleTable(PDomain, PExampleGenerator alock, int); // You can pass NULL here; this would still lock the examples
  TExampleTable(PExampleGenerator orig, int); // This int only tells that we want lock; it is not used in constructor
  bool ownsPointers;
  PExampleGenerator lock;

public:
  /* ExampleTable has some vector-like behaviour  */
        TExample &at(const int &i);
  const TExample &at(const int &i) const;
        TExample &back();
  const TExample &back() const;
  bool            empty() const;
        TExample &front();
  const TExample &front() const;
        TExample &operator[](const int &i);
  const TExample &operator[](const int &i) const;
            void reserve(const int &i);
            void growTable();
            void shrinkTable();
             int size() const;
            void erase(const int &sti);
            void erase(const int &sti, const int &eni);

            void push_back(TExample *x);
            void erase(TExample **ptr);
            void erase(TExample **fromPtr, TExample **toPtr);
            void insert(const int &sti, const TExample &ex);

  int traverse(visitproc visit, void *arg);
  int dropReferences();

  virtual TExampleIterator begin();
  bool randomExample(TExample &);

  virtual void changeDomain(PDomain);

protected:
  virtual void increaseIterator(TExampleIterator &);
  virtual void copyIterator(const TExampleIterator &, TExampleIterator &);
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &);

public:
  bool remove(TExampleIterator &it);

  TValue operator ()(const TExample &);

  virtual int  numberOfExamples();

  virtual void addExample(const TExample &);
  virtual void addExamples(PExampleGenerator orig);

  virtual bool removeExamples  (TFilter &); 
  virtual bool removeExample   (TExample &);
  virtual bool removeCompatible(TExample &);
  virtual void removeDuplicates(const int &weightID=0);
  virtual void clear();

  void sort();
  void sort(vector<int> &sortOrder);

  virtual void addMetaAttribute(const int &id, const TValue &value);
  virtual void copyMetaAttribute(const int &id, const int &source, TValue &defaultVal);
  virtual void removeMetaAttribute(const int &id);
};


/* Here for compatibility; it calls ExampleTable's constructor, telling it
   that it should not own pointers. */
class TExamplePointerTable : public TExampleTable {
public:
  __REGISTER_CLASS

  TExamplePointerTable(PDomain);
  TExamplePointerTable(PExampleGenerator orig);

  // This doesn't copy examples - the second argument serves only for locking
  TExamplePointerTable(PDomain, PExampleGenerator lock);
};


/* Returns example generator on which TExamplePointerTable can be used.
   Function simply stores examples into TExampleTable if needed */
inline PExampleGenerator fixedExamples(PExampleGenerator gen)
{ return (&*gen->begin()==&*gen->begin()) ? gen : PExampleGenerator(mlnew TExampleTable(gen)); }

/* Stores examples into TExampleTable if they are not into one already */
inline PExampleGenerator toExampleTable(PExampleGenerator gen)
{ return (gen.is_derived_from(TExampleTable) ? gen : PExampleGenerator(mlnew TExampleTable(gen))); }

#endif

