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


#include "examplegen.hpp"
#include "contingency.hpp"

#include "tdidt_stop.ppp"


bool TTreeStopCriteria::operator()(PExampleGenerator gen, const int &, PDomainContingency ocont)
{ int nor = gen->numberOfExamples();
  if ((nor==0) || (nor==1))
    return true;      // example set is too small

  char vt = gen->domain->classVar->varType;
  if (vt!=TValue::INTVAR)
    return false;  // class is continuous, may continue

  // is there more than one class left?
  if (ocont) {
    char ndcf = 0;
    TDiscDistribution const &dva=CAST_TO_DISCDISTRIBUTION(ocont->classes);
    const_ITERATE(TDiscDistribution, ci, dva)
      if ((*ci>0) && (++ndcf==2))
        return false; // at least two classes, may continue
  }
  
  else {  
    TExampleIterator ei = gen->begin();
    TValue fv = (*ei).getClass();

    while(fv.isSpecial() && ++ei)
      fv = (*ei).getClass();
    if (!ei)
      return true;

    const int fvi = fv.intV;

    while(++ei) {
      TValue &cval = (*ei).getClass();
      if (!cval.isSpecial() && (cval.intV != fvi))
        return false; // yes, may continue
    }
  }

  return true; // no, there's just one class left!
}


TTreeStopCriteria_common::TTreeStopCriteria_common(float aMaxMajor, float aMinExamples)
: maxMajority(aMaxMajor),
  minExamples(aMinExamples)
{}


TTreeStopCriteria_common::TTreeStopCriteria_common(const TTreeStopCriteria_common &old)
: TTreeStopCriteria(old),
  maxMajority(old.maxMajority),
  minExamples(old.minExamples)
{}


bool TTreeStopCriteria_common::operator()(PExampleGenerator gen, const int &weight, PDomainContingency ocont)
{ if (TTreeStopCriteria::operator()(gen, weight, ocont)) 
    return true; // inherited method says its enough

  PDistribution classDist = ocont ? ocont->classes : getClassDistribution(gen, weight);
  if (classDist->abs<minExamples)
    return true; // not enough examples

  float limit = maxMajority*classDist->abs;
  TDiscDistribution *ddva = classDist.AS(TDiscDistribution);
  if (ddva) {
    const_PITERATE(TDiscDistribution, ci, ddva)
      if (*ci>limit)
        return true;
  }
  else {
    TContDistribution *cdva = classDist.AS(TContDistribution);
    const_PITERATE(TContDistribution, ci, cdva)
      if ((*ci).second>limit)
        return true;
  }

  return false;
}

