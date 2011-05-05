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


#ifndef __TABLE_HPP
#define __TABLE_HPP

#include "examplegen.hpp"
#include "random.hpp"

class TFilter;

// Example generator which stores examples in STL's vector.
class ORANGE_API TExampleTable : public TExampleGenerator {
public:
  __REGISTER_CLASS

  TExample **examples;
  TExample **_Last, **_EndSpace;
  PRandomGenerator randomGenerator; //P random generator used by randomExample
  PExampleGenerator lock; //PR (+owner) the real owner of examples
  bool ownsExamples; //PR (+owns_instances) if false, examples in this table are references to examples in another table

  // Iterates through examples of basevector
  #define baseITERATE(x) ITERATE(vector<TExample>, x, examples)

  TExampleTable(PDomain, bool owns = true);
  TExampleTable(PExampleGenerator orig, bool owns = true); // also copies examples
  TExampleTable(PDomain, PExampleGenerator orig, bool filterMetas = false); // owns = true (cannot change domain of references); copies examples
  TExampleTable(PExampleGenerator lock, int); // owns = false; pass anything for int; this constructor locks, but does not copy
  TExampleTable(PExampleGeneratorList tables);
  ~TExampleTable();


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


  TExample &new_example();
  void delete_last();

  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  virtual TExampleIterator begin();
  bool randomExample(TExample &);

  virtual void changeDomain(PDomain, bool filterMetas = false);

protected:
  virtual void increaseIterator(TExampleIterator &);
  virtual void copyIterator(const TExampleIterator &, TExampleIterator &);
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &);

public:
  bool remove(TExampleIterator &it);

  TValue operator ()(const TExample &);

  virtual int  numberOfExamples();
  virtual float weightOfExamples(const int &weightID = 0) const;

  virtual void addExample(const TExample &, bool filterMetas = false);
  virtual void addExample(TExample *);
  virtual void addExamples(PExampleGenerator orig, bool filterMetas = false);

  virtual bool removeExamples  (TFilter &); 
  virtual bool removeExample   (TExample &);
  virtual bool removeCompatible(TExample &);
  virtual void removeDuplicates(const int &weightID=0);
  virtual void clear();

  void sort();
  void sort(vector<int> &sortOrder);

  void sortByPointers();
  
  void shuffle();
  
  virtual void addMetaAttribute(const int &id, const TValue &value);
  virtual void copyMetaAttribute(const int &id, const int &source, TValue &defaultVal);
  virtual void removeMetaAttribute(const int &id);

  virtual int checkSum(const bool includeMetas = false);
  virtual int checkSum(const bool includeMetas = false) const;
  virtual bool hasMissing() const;
  virtual bool hasMissingClass() const;
};


/* Returns example generator which can be referenced.
   Function simply stores examples into TExampleTable if needed */
inline PExampleGenerator fixedExamples(PExampleGenerator gen)
{ return (&*gen->begin()==&*gen->begin()) ? gen : PExampleGenerator(mlnew TExampleTable(gen)); }

/* Stores examples into TExampleTable if they are not into one already */
inline PExampleGenerator toExampleTable(PExampleGenerator gen)
{ return (gen.is_derived_from(TExampleTable) ? gen : PExampleGenerator(mlnew TExampleTable(gen))); }

#endif

