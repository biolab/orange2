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

#ifndef __SPEC_GEN_HPP
#define __SPEC_GEN_HPP

#include "orvector.hpp"
#include "examplegen.hpp"
#include "filter.hpp"
#include "contingency.hpp"

WRAPPER(Filter);
WRAPPER(RandomGenerator);

/*  A base for 'filter generators' i.e. generators, which can be put on top of other generator and modify its
    examples (usualy skip or add examples...). The behaviour of the generator is modified by overriding iterator
    handling method (usualy begin and increaseIterator). */
class ORANGE_API TAdapterGenerator : public TExampleGenerator {
public:
  __REGISTER_CLASS

  /*  Iterators, pointing to the first and the one-beyond-the-last example from the underlying generator.
      They are not necessarily equal to gen->begin() and gen->end() so the TAdapterGenerator can be used to
      select a set of consecutive examples of underlying generator. */
  TExampleIterator first, last;

  TAdapterGenerator(PDomain, const TExampleIterator &first, const TExampleIterator &last);
  TAdapterGenerator(PDomain, PExampleGenerator);
  TAdapterGenerator(PExampleGenerator);

  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  virtual TExampleIterator begin();
  virtual TExampleIterator begin(void *derData);
  virtual bool randomExample(TExample &);

  virtual int numberOfExamples();

  virtual void increaseIterator(TExampleIterator &);
  virtual bool sameIterators(const TExampleIterator &, const TExampleIterator &);
  virtual void deleteIterator(TExampleIterator &);
  virtual void copyIterator(const TExampleIterator &, TExampleIterator &);
};


class ORANGE_API TAdapterIteratorData {
public:
  TExampleIterator subIterator;
  void *data;

  TAdapterIteratorData(const TExampleIterator &, void * =NULL);
};

/*  Derived from TAdapterGenerator, this class overrides the begin() and increaseIterator(void *) methods to
    skip the examples which are not accepted by the given filter. */
class ORANGE_API TFilteredGenerator : public TAdapterGenerator {
public:
  __REGISTER_CLASS

  PFilter filter; //P decides which examples are skipped
  
  TFilteredGenerator(PFilter, PDomain, const TExampleIterator &, const TExampleIterator &);
  TFilteredGenerator(PFilter, PExampleGenerator);

  virtual TExampleIterator begin();
  virtual void increaseIterator(TExampleIterator &);
};


WRAPPER(EFMDataDescription)


/*  Changes the example someway by redefining begin and increaseIterator to call an abstract
    method changeExample */
class ORANGE_API TChangeExampleGenerator : public TAdapterGenerator {
public:
  __REGISTER_ABSTRACT_CLASS

  TChangeExampleGenerator(PDomain, const TExampleIterator &, const TExampleIterator &);
  TChangeExampleGenerator(PExampleGenerator);

  virtual TExampleIterator begin();
  virtual void increaseIterator(TExampleIterator &);

  virtual TExampleIterator changeExample(const TExampleIterator &)=0;
};


/*  Derived from TChangeExampleGenerator, TMissValuesGenerator replaces values of certain
    attributes (given the probability for change) with DK or DC */
class ORANGE_API TMissValuesGenerator : public TChangeExampleGenerator {
public:
  __REGISTER_CLASS

  PIntFloatList replaceProbabilities; //P probabilities for replacing attributes' values
  PRandomGenerator randomGenerator; //P random generator

  TMissValuesGenerator(const vector<pair<int, float> > &, PDomain &, TExampleIterator &, TExampleIterator &);
  TMissValuesGenerator(const vector<pair<int, float> > &, PExampleGenerator);

  TExampleIterator changeExample(const TExampleIterator &it);
};


class ORANGE_API TNoiseValuesGenerator : public TChangeExampleGenerator {
public:
  __REGISTER_CLASS

  PIntFloatList replaceProbabilities; //P probabilities for replacing attributes' values
  PRandomGenerator randomGenerator; //P random generator

  TNoiseValuesGenerator(const vector<pair<int, float> > &, PDomain &, TExampleIterator &, TExampleIterator &);
  TNoiseValuesGenerator(const vector<pair<int, float> > &, PExampleGenerator);

  TExampleIterator changeExample(const TExampleIterator &it);
};


class ORANGE_API TGaussianNoiseGenerator : public TChangeExampleGenerator {
public:
  __REGISTER_CLASS

  PIntFloatList deviations; //P deviations for attributes' values
  PRandomGenerator randomGenerator; //P random generator

  TGaussianNoiseGenerator(const vector<pair<int, float> > &, PDomain &, TExampleIterator &, TExampleIterator &, PRandomGenerator = PRandomGenerator());
  TGaussianNoiseGenerator(const vector<pair<int, float> > &, PExampleGenerator, PRandomGenerator = PRandomGenerator());

  TExampleIterator changeExample(const TExampleIterator &it);
};

#endif
