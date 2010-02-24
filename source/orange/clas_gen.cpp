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

#include "stladdon.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "classify.hpp"
#include "examplegen.hpp"

#include "clas_gen.ppp"


// Default constructor; domain must be set later.
TClassifierGenerator::TClassifierGenerator()
: TExampleGenerator(),
  classify()
{}


// Constructor, given a domain. Classifier must be set later.
TClassifierGenerator::TClassifierGenerator(PDomain &dom)
: TExampleGenerator(dom),
  classify()
{}


// Constructor, given a classifier. Domain is the same as classifier's.
TClassifierGenerator::TClassifierGenerator(PDomain &dom, PClassifier &clsf)
: TExampleGenerator(dom),
  classify(clsf)
{}


// Returns the class of example, if classify!=NULL; DK otherwise.
TValue TClassifierGenerator::operator ()(const TExample &ex)
{ return classify ? classify->operator()(ex) : domain->classVar->DK(); }



/*  Constructs an example with random values of attributes (as obtained by randomValue() method of TVariable),
    adding the class, obtained from 'classify'. This example generator should be mostly used with this method
    and not by iterating through (possibly very big number of) examples. */
bool TClassifierGenerator::randomExample(TExample &example)
{ TExample::iterator ei=example.begin();
  PITERATE(TVarList, di, domain->attributes) 
    *(ei++)=(*di)->randomValue();
  if (domain->classVar)
    *ei=operator()(example);
  return 1;
}

/*  Describes the number of examples. It returns the product of noOfValues for all attributes if it is defined
    for all the attributes. Otherwise, if all the variables have the firstValue() method (which is required for
    this generator to work, anyway) it returns NOEX_INFINITE, and NOEX_DONT_KNOW otherwise. */
int TClassifierGenerator::numberOfExamples()
{ TVarList::iterator vi, ei=domain->attributes->end();
  
  int prod=1;
  for(vi=domain->attributes->begin(), ei--; (vi!=ei); prod*= (*(vi++))->noOfValues());
  if (prod)
    return prod;

  TValue foo;
  while((vi!=ei) && ((*(vi++))->firstValue(foo)));
  if (vi==ei) return NOEX_INFINITE;

  return NOEX_DONT_KNOW;
}

// Constructs an iterator. Attributes are initialized by firstValue() and class by classifying using 'classify'.
TExampleIterator TClassifierGenerator::begin()
{ TExampleIterator ri(domain, this);
  TExample::iterator ei(ri.privateExample.begin());
  PITERATE(TVarList, di, domain->attributes)
    if (!(*di)->firstValue(*(ei++))) {
      deleteIterator(ri);
      break;
    }
  if (domain->classVar)
    if (ri.example)
      ri.privateExample.setClass(operator()(ri.privateExample));
  return ri;
}


/*  Increases the values of attributes. The last attribute's value is increased first; if it was the last value,
    it is reinitialized to the firstValue() and the next attribute's value is increased. If all the values are
    the last possible values, iterator is set to end() */
void TClassifierGenerator::increaseIterator  (TExampleIterator &i)
{ TExample::iterator ri=(*i).begin()+domain->attributes->size()-1;
  TVarList::reverse_iterator vi=domain->variables->rbegin();
  while((vi!=domain->variables->rend()) && !(*vi)->nextValue(*ri))
    (*(vi++))->firstValue(*(ri--));
  if (vi==domain->variables->rend())
    deleteIterator(i);
  (*i).setClass(operator()(*i));
}


void TClassifierGenerator::deleteIterator(TExampleIterator &data)
{}

void TClassifierGenerator::copyIterator(const TExampleIterator &src, TExampleIterator &dest)
{ dest = src; }


// Compares the examples that pointers point to
bool TClassifierGenerator::sameIterators(const TExampleIterator &data1, const TExampleIterator &data2)
{ return data1.privateExample == data2.privateExample; }




/* ******* TClassifierRandomGenerator *****/


// Default constructor; domain must be set later.
TClassifierRandomGenerator::TClassifierRandomGenerator()
: TExampleGenerator(),
  classify(),
  noOfEx(-1)
{}


// Constructor, given a domain. Classifier must be set later.
TClassifierRandomGenerator::TClassifierRandomGenerator(PDomain &dom, int anoOfEx)
: TExampleGenerator(dom),
  classify(),
  noOfEx(anoOfEx)
{}


// Constructor, given a classifier. Domain is the same as classifier's.
TClassifierRandomGenerator::TClassifierRandomGenerator(PDomain &dom, PClassifier &clsf, int anoOfEx)
: TExampleGenerator(dom),
  classify(clsf),
  noOfEx(anoOfEx)
{}


/*  Constructs an example with random values of attributes (as obtained by randomValue() method of TVariable),
    adding the class, obtained from 'classify'. This example generator should be mostly used with this method
    and not by iterating through (possibly very big number of) examples. */
bool TClassifierRandomGenerator::randomExample(TExample &ex)
{ TExample::iterator ei=ex.begin();
  PITERATE(TVarList, di, domain->attributes)
    *(ei++)=(*di)->randomValue();
  ex.setClass(classify ? classify->operator()(ex) : domain->classVar->DK());
  return true;
}


// Constructs an iterator. Example is initialized by randomExample().
TExampleIterator TClassifierRandomGenerator::begin()
{ if (!noOfEx) return TExampleIterator(this);

  TExampleIterator it(domain, this, mlnew int(0));
  randomExample(it.privateExample);

  return it;
}


// Returns NOEX_INFINITE
int TClassifierRandomGenerator::numberOfExamples()
{ return noOfEx<0 ? NOEX_INFINITE : noOfEx; }


// Constructs a new example by randomExample()
void TClassifierRandomGenerator::increaseIterator  (TExampleIterator &i)
{ int &ec=*(int *)(i.data);
  if ((ec!=-1) && (++ec == noOfEx))
    deleteIterator(i);
  else randomExample(i.privateExample);
}


// Compares the examples that pointers point to
bool TClassifierRandomGenerator::sameIterators(const TExampleIterator &data1, const TExampleIterator &data2)
{ return false; }


// Copies the copyIterator method of the underlying generator
void TClassifierRandomGenerator::copyIterator(const TExampleIterator &src, TExampleIterator &dest)
 { TExampleGenerator::copyIterator(src, dest);
   dest.data=mlnew int(*(int *)(src.data));
 }

void TClassifierRandomGenerator::deleteIterator(TExampleIterator &it)
{  mldelete (int *)(it.data);
   TExampleGenerator::deleteIterator(it);
}
