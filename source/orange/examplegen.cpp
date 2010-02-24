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


#include "vars.hpp"
#include "domain.hpp"

#include "crc.h"

#include "examplegen.ppp"

DEFINE_TOrangeVector_classDescription(PExampleGenerator, "TExampleGeneratorList", true, ORANGE_API)


int generatorVersion=0;

TExampleGenerator::TExampleGenerator()
: domain(),
  version(-1)
{}

TExampleGenerator::TExampleGenerator(PDomain dom)
: domain(dom),
  version(-1)
{}

int TExampleGenerator::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);

/* No need for this: iterators have just pointers, not references to generators!
  if (!myIterators.empty())
    myIterators.front()->generator.mark(myIterators.size());
*/
  const_ITERATE(list<iterator *>, ii, myIterators)
    if ((*ii)->example==&((*ii)->privateExample))
      TRAVERSE((*ii)->privateExample.traverse);
  return 0;
}


int TExampleGenerator::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);

  while (myIterators.size())
    deleteIterator(*myIterators.front());
  return 0;
}


void TExampleGenerator::copyIterator(const TExampleIterator &, TExampleIterator &)
{}

void TExampleGenerator::deleteIterator(TExampleIterator &it)
{ it.example = NULL; 
  it.data = NULL;
}

// Returns an iterator pointing past the last example
TExampleIterator TExampleGenerator::end()
{ return TExampleIterator(this); }


void TExampleGenerator::examplesHaveChanged()
{ version = ++generatorVersion; }


#define NOT_SUPPORTED(method) \
{ raiseError("method '"#method"' is not supported"); }

void TExampleGenerator::addMetaAttribute(const int &, const TValue &)
NOT_SUPPORTED(addMetaAttribute)

void TExampleGenerator::copyMetaAttribute(const int &, const int &, TValue &)
NOT_SUPPORTED(copyMetaAttribute)

void TExampleGenerator::removeMetaAttribute(const int &)
NOT_SUPPORTED(removeMetaAttribute)


int TExampleGenerator::checkSum(const bool includeMetas)
{ unsigned long crc;
  INIT_CRC(crc);
  for(TExampleIterator ei(begin()); ei; ++ei)
    (*ei).addToCRC(crc, includeMetas);
  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}


float TExampleGenerator::weightOfExamples(const int &weightID) const
{
  raiseError("cannot compute the weight of examples");
  return 0.0;
}


// Constructs the iterator, setting generator and data fields to the given values.
TExampleIterator::TExampleIterator(TExampleGenerator *agen, TExample *anexam, void *adata) 
: generator(agen),
  example(anexam),
  data(adata) 
{ if (generator)
    generator->myIterators.push_front(this);
}


TExampleIterator::TExampleIterator(const TExample &anexam, TExampleGenerator *agen, void *adata) 
: generator(agen),
  example(&privateExample),
  data(adata),
  privateExample(anexam)
{ if (generator)
    generator->myIterators.push_front(this);
}

TExampleIterator::TExampleIterator(PDomain domain, TExampleGenerator *agen, void *adata) 
: generator(agen), example(&privateExample), data(adata), privateExample(domain)
{ if (generator)
    generator->myIterators.push_front(this);
}

TExampleIterator::TExampleIterator(const TExampleIterator &other)
: generator(other.generator),
  example(other.example==&other.privateExample ? &privateExample : other.example),
  data(NULL),
  privateExample(other.example==&other.privateExample ? other.privateExample : TExample())
{ if (generator) {
    generator->copyIterator(other, *this); 
    generator->myIterators.push_front(this);
  }
}


int TExampleIterator::traverse(visitproc visit, void *arg) const
{ /* Iterator has an unwrapped pointer to generator.
     We would have to call generator->traverse if it would not be wrapped.
     However, all generators are expected to be wrapped (sooner or later)
     so this won't cause a permanent memory leak. */
   return (example==&privateExample) ? example->traverse(visit, arg) : 0;
}


int TExampleIterator::dropReferences()
{ return (example==&privateExample) ? example->dropReferences() : 0;
}


TExampleIterator::~TExampleIterator()    
{ 
  if (generator) {
    generator->deleteIterator(*this); 
    generator->myIterators.remove(this);
  }
}


TExampleIterator &TExampleIterator::operator =(const TExampleIterator &other)
{ 
  if (other.example==&other.privateExample) {
    privateExample = other.privateExample;
    example = &privateExample;
  }
  else {
    privateExample = TExample();
    example = other.example;
  }

  if (generator != other.generator) {
    if (generator) {
      generator->deleteIterator(*this);
      generator->myIterators.remove(this);
    }
    
    generator = other.generator;
    generator->copyIterator(other, *this);
    generator->myIterators.push_back(this);
  }

  return *this;
}


bool TExampleIterator::operator == (const TExampleIterator &other)
{ return (generator==other.generator) && (generator->sameIterators(*this, other)); }

bool TExampleIterator::operator != (const TExampleIterator &other)
{ return (generator!=other.generator) || (!generator->sameIterators(*this, other)); }

