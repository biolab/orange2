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


#include <limits>
#include "stladdon.hpp"

#include "errors.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "filter.hpp"
#include "distvars.hpp"
#include "preprocess.hpp"


#include "stat.hpp"

#include "spec_gen.ppp"

/* ******* TAdapterGenerator *****/


/*  A constructor, given domain and pair of iterators. This constructor extracts gen field from iterators, which
    should generally be avoided. DispIt is false and the iterators are not deleted when generator is destructed. */
TAdapterGenerator::TAdapterGenerator(PDomain dom, const TExampleIterator &afirst, const TExampleIterator &alast)
  : TExampleGenerator(dom), first(afirst), last(alast)
{};


// A constructor, given an underlying generator. First and last are initialized by gen->begin() and gen->end()
TAdapterGenerator::TAdapterGenerator(PExampleGenerator gen)
  : TExampleGenerator(gen->domain), first(gen->begin()), last(gen->end())
{};


// A constructor, given a (new) domain and an underlying generator. First and last are initialized by gen->begin() and gen->end()
TAdapterGenerator::TAdapterGenerator(PDomain dom, PExampleGenerator gen)
  : TExampleGenerator(dom), first(gen->begin()), last(gen->end())
{};



// Returns first.
TExampleIterator TAdapterGenerator::begin()
{ if (first) 
    return TExampleIterator(TExample(domain, *first), this, mlnew TAdapterIteratorData(TExampleIterator(first)));
  else
    return TExampleIterator(this, NULL, mlnew TAdapterIteratorData(TExampleIterator(first)));
}

/*  Returns first, acceptint additional void pointer passed from a derived class;
    this method is therefore to be called from derived class begin() methods */
TExampleIterator TAdapterGenerator::begin(void *derData)
{ if (first) 
    return TExampleIterator(TExample(domain, *first), this, mlnew TAdapterIteratorData(TExampleIterator(first), derData));
  else
    return TExampleIterator(this, NULL, mlnew TAdapterIteratorData(TExampleIterator(first), derData));
}

/*  Returns false. Note that gen->randomExample() cannot be called since the TAdapterGenerator might not know how
    to transform it, If a derived adapter is capable of adapting random examples, this method should be implemented. */
bool TAdapterGenerator::randomExample(TExample &)
{ return false; }


/*  Returns NOEX_TRACTABLE if underlying generator returns the exact number; otherwise it returns the same number
    as underlying generator. If the deriverd adapter can do better, it should redefine this method. */
int  TAdapterGenerator::numberOfExamples()
{ int noi=first.generator->numberOfExamples(); 
  return (noi>0) ? NOEX_TRACTABLE : noi;
}


#define AD_cast(x) ((TAdapterIteratorData *&)((x).data))


int TAdapterGenerator::traverse(visitproc visit, void *arg)
{ // deliberately skipping TExampleGenerator --- I FORGOT THE REASON WHY!
  TRAVERSE(TOrange::traverse)
  TRAVERSE(first.traverse)
  TRAVERSE(last.traverse);

/* No need for this: iterators have just pointers, not references to generators!
  if (!myIterators.empty())
    myIterators.front()->generator.mark(myIterators.size());
*/
  const_ITERATE(list<TExampleIterator *>, ii, myIterators) {
    if ((*ii)->example==&((*ii)->privateExample))
      TRAVERSE((*ii)->privateExample.traverse);
    TRAVERSE(AD_cast(**ii)->subIterator.traverse);
  }
  return 0;
}


int TAdapterGenerator::dropReferences()
{ // deliberately skipping TExampleGenerator --- I FORGOT THE REASON WHY!
  DROPREFERENCES(first.dropReferences)
  DROPREFERENCES(last.dropReferences);
  DROPREFERENCES(TExampleGenerator::dropReferences)
  return 0;
}


// Calls the increaseIterator of the underlying generator but sets the iterator to end if it reaches last.
void TAdapterGenerator::increaseIterator(TExampleIterator &i)                        
 { if (i.example) {
     TExampleIterator &subIterator=AD_cast(i)->subIterator;
     if (++subIterator) 
       i.privateExample=TExample(domain, *subIterator);
     else
       deleteIterator(i);
   }
 }

// Calls the sameIterators method of the underlying generator
bool TAdapterGenerator::sameIterators(const TExampleIterator &i1, const TExampleIterator &i2)
 { return (!i1 && !i2) || (i1 && i2 && (AD_cast(i1)->subIterator == AD_cast(i2)->subIterator)); }

// Calls the deleteIterator method of the underlying generator
void TAdapterGenerator::deleteIterator(TExampleIterator &i)
 { mldelete AD_cast(i);
   AD_cast(i)=NULL;
   TExampleGenerator::deleteIterator(i); }

// Calls the copyIterator method of the underlying generator
void TAdapterGenerator::copyIterator(const TExampleIterator &src, TExampleIterator &dest)
 {  TExampleGenerator::copyIterator(src, dest);
    AD_cast(dest)=mlnew TAdapterIteratorData(AD_cast(src)->subIterator, NULL);
 }



TAdapterIteratorData::TAdapterIteratorData(const TExampleIterator &oi, void *dt)
 : subIterator(oi), data(dt)
 {}

/* ******* TFilteredGenerator *****/

TFilteredGenerator::TFilteredGenerator(PFilter afilter, PDomain dom, const TExampleIterator &afirst, const TExampleIterator &alast)
  : TAdapterGenerator(dom, afirst, alast), filter(afilter)
{};


TFilteredGenerator::TFilteredGenerator(PFilter afilter, PExampleGenerator gen)
  : TAdapterGenerator(gen), filter(afilter)
{};


TExampleIterator TFilteredGenerator::begin()
{ TExampleIterator i(TAdapterGenerator::begin());
  for(; i && !filter->operator()(*i); TAdapterGenerator::increaseIterator(i));
  return i;
}


// Increases the iterator until it reaches an example that the filter lets through.
void TFilteredGenerator::increaseIterator(TExampleIterator &i)
{ for(TAdapterGenerator::increaseIterator(i); i && !filter->operator()(*i); TAdapterGenerator::increaseIterator(i)); }



TChangeExampleGenerator::TChangeExampleGenerator(PDomain dom, const TExampleIterator &af, const TExampleIterator &al)
 : TAdapterGenerator(dom, af, al) {}

TChangeExampleGenerator::TChangeExampleGenerator(PExampleGenerator gen)
 : TAdapterGenerator(gen) {}


TExampleIterator TChangeExampleGenerator::begin()
{ return changeExample(TAdapterGenerator::begin()); }


// Increases the iterator until it reaches an example that the filter lets through.
void TChangeExampleGenerator::increaseIterator(TExampleIterator &i)
{ TAdapterGenerator::increaseIterator(i);
  if (i) changeExample(i);
}



TMissValuesGenerator::TMissValuesGenerator(const vector<float> &rp, PDomain &dom, TExampleIterator &afirst, TExampleIterator &alast, PRandomGenerator rgen)
 : TChangeExampleGenerator(dom, afirst, alast), replaceProbabilities(mlnew TFloatList(rp)), randomGenerator(rgen)
 {}

TMissValuesGenerator::TMissValuesGenerator(const vector<float> &rp, PExampleGenerator gen, PRandomGenerator rgen)
 : TChangeExampleGenerator(gen), replaceProbabilities(mlnew TFloatList(rp)), randomGenerator(rgen)
 {}


TExampleIterator TMissValuesGenerator::changeExample(const TExampleIterator &it)
{ if (it) {
    TExample::iterator ei(it.example->begin());
    const_PITERATE(TFloatList, pi, replaceProbabilities) {
      if (*pi<0) {
        if ((randomGenerator ? randomGenerator->randfloat() : randfloat()) < -*pi)
          (*ei).setDC();
      }
      else if (*pi>0) {
        if ((randomGenerator ? randomGenerator->randfloat() : randfloat()) <  *pi)
          (*ei).setDK();
      }
      ei++;
    }
  }
  return it;
}


TNoiseValuesGenerator::TNoiseValuesGenerator(const vector<float> &rp, PDomain &dom, TExampleIterator &afirst, TExampleIterator &alast, PRandomGenerator rgen)
 : TChangeExampleGenerator(dom, afirst, alast), replaceProbabilities(mlnew TFloatList(rp)), randomGenerator(rgen)
 { TVarList::const_iterator vi(domain->variables->begin()); 
   PITERATE(TFloatList, pi, replaceProbabilities)
     if ((*vi)->noOfValues()<2)
       *pi=0;
 }

TNoiseValuesGenerator::TNoiseValuesGenerator(const vector<float> &rp, PExampleGenerator gen, PRandomGenerator rgen)
 : TChangeExampleGenerator(gen), replaceProbabilities(mlnew TFloatList(rp)), randomGenerator(rgen) {}


TExampleIterator TNoiseValuesGenerator::changeExample(const TExampleIterator &it)
{ if (it) {
    TExample::iterator ei(it.example->begin());
    TVarList::iterator vi(domain->variables->begin());
    PITERATE(TFloatList, pi, replaceProbabilities) {
      if ((*pi>0) && ((randomGenerator ? randomGenerator->randfloat() : randfloat()) < *pi))
        if (((*ei)=(*vi)->randomValue(randomGenerator ? randomGenerator->randint() : randint())).isDC())
          raiseError("attribute '%s' cannot give randomValues.", (*vi)->name.c_str());
      ei++;
      vi++;
    }
  }
  return it;
}


TGaussianNoiseGenerator::TGaussianNoiseGenerator(const vector<float> &rp, PDomain &dom, TExampleIterator &afirst, TExampleIterator &alast, PRandomGenerator rgen)
 : TChangeExampleGenerator(dom, afirst, alast), deviations(mlnew TFloatList(rp)), randomGenerator(rgen)
 {}

TGaussianNoiseGenerator::TGaussianNoiseGenerator(const vector<float> &rp, PExampleGenerator gen, PRandomGenerator rgen)
 : TChangeExampleGenerator(gen), deviations(mlnew TFloatList(rp)), randomGenerator(rgen) {}


class genrandfloat_11 {
public:
  PRandomGenerator rgen;
  genrandfloat_11(PRandomGenerator agen=PRandomGenerator()) : rgen(agen) {}
  double operator()(const double &x, const double &y) { return rgen ? rgen->randfloat(x, y) : randfloat(x, y); }
};

TExampleIterator TGaussianNoiseGenerator::changeExample(const TExampleIterator &it)
{ if (it) {
    TExample::iterator ei(it.example->begin());
    TVarList::iterator vi(domain->variables->begin());
    PITERATE(TFloatList, pi, deviations) {
      if ((*pi>0.0) && !(*ei).isSpecial() && ((*vi)->varType==TValue::FLOATVAR))
        *ei = TValue(gasdev(float(*ei), *pi, genrandfloat_11(randomGenerator)));
      ei++;
      vi++;
    }
  }
  return it;
}

#undef AD_cast


