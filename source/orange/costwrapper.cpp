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


#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "cost.hpp"

#include "costwrapper.ppp"

TCostWrapperLearner::TCostWrapperLearner(PCostMatrix cm, PLearner bl)
: basicLearner(bl),
  costMatrix(cm)
{}


PClassifier TCostWrapperLearner::operator ()(PExampleGenerator gen, const int &weight)
{ return mlnew TCostWrapperClassifier(costMatrix, basicLearner->operator()(gen, weight)); }


TCostWrapperClassifier::TCostWrapperClassifier(PCostMatrix cm, PClassifier bc)
: TClassifier(bc->classVar, false),
  classifier(bc),
  costMatrix(cm)
{}


PDiscDistribution TCostWrapperClassifier::getRisks(const TExample &ex)
{ return getRisks(classifier->classDistribution(ex)); }


PDiscDistribution TCostWrapperClassifier::getRisks(PDistribution wdval)
{ const TDiscDistribution &dval = CAST_TO_DISCDISTRIBUTION(wdval);
  PDiscDistribution risks = mlnew TDiscDistribution;

  for(int predicted=0, dsize = dval.size(); predicted<dsize; predicted++) {
    float thisCost=0;
    for(int correct=0; correct<dsize; correct++)
      thisCost += dval[correct] * costMatrix->cost(predicted, correct);
    risks->push_back(thisCost);
  }
  return risks;
}


TValue TCostWrapperClassifier::operator ()(const TExample &ex)
{ return operator()(getRisks(ex)); }


TValue TCostWrapperClassifier::operator ()(PDiscDistribution risks)
{ float ccost = numeric_limits<float>::max();
  int wins=0, bestPrediction;
  const_ITERATE(TDiscDistribution, ri, risks.getReference())
    if (   (*ri<ccost)  && ((wins=1)==1)
        || (*ri==ccost) && globalRandom->randbool(++wins)) {
      bestPrediction=ri-risks->begin();
      ccost=*ri;
    }
   return wins ? TValue(bestPrediction) : classVar->DK();
}
