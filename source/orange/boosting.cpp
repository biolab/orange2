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


#include <math.h>
#include <iostream>
#include "stladdon.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "meta.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "boosting.ppp"


DEFINE_TOrangeVector_classDescription(PWeight_Classifier, "Weight_ClassifierList")


TBoostLearner::TBoostLearner(int aT)
: weakLearner(),
  T(aT)
{}

 
PClassifier TBoostLearner::operator()(PExampleGenerator ogen, const int &weight)
{ if (!ogen->domain->classVar)
    raiseError("class-less domain");

  if (ogen->domain->classVar->varType!=TValue::INTVAR)
    raiseError("discrete class expected");

  PExampleGenerator gen = toExampleTable(ogen);
 
  TVotingClassifier *classifier = mlnew TVotingClassifier;
  PClassifier wclassifier = classifier;

  float weights = 0.0;
  long myWeight = getMetaID();

  if (weight) {
   { PEITERATE(ei, gen) 
       weights += float((*ei).meta[weight]); }
   { PEITERATE(ei, gen)
       (*ei).meta.setValue(myWeight, TValue(float((*ei).meta[weight])/weights)); }
  }
  else {
    weights = gen->numberOfExamples();
    PEITERATE(ei, gen)
      (*ei).meta.setValue(myWeight, TValue(float(1.0)/weights));
  }

  vector<bool> correct(gen->numberOfExamples());
 
  for(int t = 0; t<T; t++) {
    PClassifier newClassifier = weakLearner->operator()(gen, myWeight);
    float epsilon = 0.0;
    { vector<bool>::iterator bi(correct.begin());
      PEITERATE(ei, gen) {
        TValue predicted = newClassifier->operator()(*ei);
        *bi = (!predicted.isSpecial() && (predicted==(*ei).getClass()));
        if (!*bi)
          epsilon += float((*ei).meta[myWeight]);
        bi++;
      }
    }
    if (epsilon>0.5) {
      if (!t)
        raiseError("failed, first iteration error, %5.3f, is larger than 0.5", epsilon);
      break;
    }

    float beta = epsilon/(1-epsilon);
    classifier->classifiers->push_back(mlnew TWeight_Classifier(log(1/beta), newClassifier));

    float Z = 0.0;
    { vector<bool>::iterator bi(correct.begin());
      PEITERATE(ei, gen) {
        if (*(bi++))
          (*ei).meta[myWeight].floatV *= beta;
        Z += (*ei).meta[myWeight].floatV;
      }
    }

    PEITERATE(ei, gen)
      (*ei).meta[myWeight].floatV/=Z;
  }

  PEITERATE(ei, gen)
    (*ei).meta.removeValue(myWeight);

  return wclassifier;
}

 
TWeight_Classifier::TWeight_Classifier(const float &aw, PClassifier acl)
: weight(aw),
  classifier(acl)
{}


PDistribution TVotingClassifier::classDistribution(const TExample &ex)
{ PDistribution wsum = TDistribution::create(classVar);
  TDiscDistribution &sum = const_cast<TDiscDistribution &>(CAST_TO_DISCDISTRIBUTION(wsum));

  PITERATE(TWeight_ClassifierList, ci, classifiers)
    sum.mul((*ci)->classifier->classDistribution(ex), (*ci)->weight);
  sum.normalize();
  return wsum;
}
