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


#ifndef __EXAMPLE_GEN_HPP
#define __EXAMPLE_GEN_HPP

#include <list>
#include "root.hpp"
#include "domain.hpp"

using namespace std;

const int
  NOEX_DONT_KNOW=-1,   
  /* number of examples is small enough to be explored */                       
  NOEX_TRACTABLE=-2, 
  /* there's a lot of examples; randomExample is recommended if it exists */    
  NOEX_FINITE=-3,
  /* iterators might not be useful since they would generate an infinite number of examples */
  NOEX_INFINITE=-4;



class ORANGE_API TExampleIterator;

extern int generatorVersion;

/*  A base class for objects that 'generate' examples, which can be traversed using iterators.
    Iterators that point to the first example and beyond the last example are returned by
    begin() and end() methods. We recommend to use them with copy constructors, for example
    for(TExampleTable::iterator i(table.begin()); i!=table.end(); i++)
    and not
    for(TExampleTable::iterator i=table.begin(); ...
    The second form can be inefficient or even cause problems under some compilers, if
    iterator handling methods are not written well. */

class ORANGE_API TExampleGenerator : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  PDomain domain; //P domain
  int version; //PR unique version identifier

  typedef TExampleIterator iterator;

  #ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
  #endif

  list<iterator *> myIterators;

  #ifdef _MSC_VER
    #pragma warning(pop)
  #endif

  TExampleGenerator();
  TExampleGenerator(PDomain dom);

  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  virtual TExampleIterator begin() =0;
  virtual TExampleIterator end();

  virtual bool randomExample(TExample &) =0;

  /*  An abstract method which returns the number or estimate of number of examples.
      That is, when exact number is not known, it can return NOEX_TRACTABLE (if number of examples is 
      known to be 'small', for instance when examples are read from the file), NOEX_FINITE (when number
      of examples is finite but can be large -- for instance, when the example space is completely covered),
      NOEX_INFINITE (when the number is known to be infinite) or NOEX_DONT_KNOW when nothing can be said about it. */
  virtual int numberOfExamples() =0;

  virtual float weightOfExamples(const int &weightID = 0) const;

  /*  Iterators handling methods should mostly be defined in derived classes. Methods
      increaseIterator and sameIterators are abstract, while deleteIterator does nothing and
      'copyIterator' throws an error saying that 'Iterators of this type cannot be copied.  */
  virtual void increaseIterator(TExampleIterator &)=0;
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &)=0;
  virtual void deleteIterator(TExampleIterator &);
  virtual void copyIterator(const TExampleIterator &source, TExampleIterator &dest);

  void examplesHaveChanged();

  virtual void addMetaAttribute(const int &id, const TValue &value);
  virtual void copyMetaAttribute(const int &id, const int &source, TValue &defaultVal);
  virtual void removeMetaAttribute(const int &id);

  virtual int checkSum(const bool includeMetas=false);
};


WRAPPER(ExampleGenerator);

#define TExampleGeneratorList TOrangeVector<PExampleGenerator> 
VWRAPPER(ExampleGeneratorList)


#include "examples.hpp"

/*  TExampleIterator is a pointer-like object, with similar functionality as iterators in STL.
    It can be dereferenced to get an example, increased (to point to the next examples), and compared
    with other examples (to see if they point to the same example). 
    It is used to access (by iterating through) the examples in a generator. */
class TExampleIterator {
public:
  // A pointer to the corresponding generator. Although public, the use this field should be avoided, if possible.
  TExampleGenerator *generator;
  /*  An example that iterator points to; this can be a pointer to a copy stored in generator or to privateExample.
      The only situation in which this would point to a specially allocated copy in memory would be when it needs to point
      to an instance of a class derived from TExample (such as TExampleForMissing).
      If NULL, iterator points beyond the last example (i.e. equals end()). */
  TExample *example;
  // Used by the generator to store the additional data it needs
  void *data;

  TExample privateExample;

  // Constructs the iterator, setting generator and data fields to the given values.
  TExampleIterator(TExampleGenerator *agen=NULL, TExample *anexam=NULL, void *adata =NULL);
  TExampleIterator(const TExample &anexam, TExampleGenerator *agen=NULL, void *adata =NULL);
  TExampleIterator(PDomain domain, TExampleGenerator *agen=NULL, void *adata =NULL);
  TExampleIterator::TExampleIterator(const TExampleIterator &other);
  int traverse(visitproc visit, void *arg) const;
  int dropReferences();
  ~TExampleIterator();

  TExampleIterator &operator =(const TExampleIterator &other);

  bool operator == (const TExampleIterator &other);
  bool operator != (const TExampleIterator &other);

  inline TExampleIterator &TExampleIterator::operator ++ ()
  { if (!example)
      raiseErrorWho("exampleIterator", "out of range");
    generator->increaseIterator(*this);
    return *this;
  }

  inline TExample &operator *() const
  { return *example; }

  inline operator bool() const
  { return example!=NULL; }
};


/* A macro for iterating through the examples of generator */
#define EITERATE(it,co) for(TExampleIterator it((co).begin()); it; ++it)
#define PEITERATE(it,co) for(TExampleIterator it((co)->begin()); it; ++it)

#endif
