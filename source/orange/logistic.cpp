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

    Authors: Martin Mozina, Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/

#include "examples.hpp"
#include "classify.hpp"
#include "logistic.ppp"
#include "math.h"


TLogisticLearner::TLogisticLearner() 
{}


PClassifier TLogisticLearner::operator()(PExampleGenerator gen, const int &weight)
{ 
  
   // check for class variable	
  if (!gen->domain->classVar)
    raiseError("class-less domain");
  // class has to be discrete!
  if (gen->domain->classVar->varType != TValue::INTVAR)
    raiseError("discrete class attribute expected");
  // attributes have to be continuous 
  PITERATE(TVarList, vli, gen->domain->attributes) {
	  if ((*vli)->varType == TValue::INTVAR) 
	    raiseError("only continuous attributes expected");
  }

  // construct result classifier	
  TLogisticClassifier *lrc = mlnew TLogisticClassifier(gen->domain);
  PClassifier cl = lrc;

  // construct a LR fitter
  fitter = PLogisticFitter(mlnew TLogisticFitterMinimization());

  // fit logistic regression 
  lrc->beta = fitter->call(gen, lrc->beta_se);

  // return classifier containing domain, beta and standard errors of beta 
  return cl;
}


TLogisticClassifier::TLogisticClassifier() 
{}


TLogisticClassifier::TLogisticClassifier(PDomain dom) 
: TClassifierFD(dom, true)
{};


PDistribution TLogisticClassifier::classDistribution(const TExample &origexam)
{   
	// domain has to exist, otherwise construction
	// of new example is impossible
	checkProperty(domain);

	// construct new example from domain & original example
	TExample example(domain, origexam);


	// multiply example with beta
	float *prob = new float[2];
	TFloatList::const_iterator b(beta->begin()), be(beta->end());
	TExample::const_iterator ei(example.begin()), ee(example.end());
	TVarList::const_iterator vi(example.domain->attributes->begin());

	// get beta 0
	prob[1] = *b;
	b++;
	// multiply beta with example
	for (; (b!=be) && (ei!=ee); ei++, b++, vi++) {
		if ((*ei).isSpecial())
			raiseError("unknown value in attribute '%s'", (*vi)->name.c_str());
		prob[1] += (*ei).floatV*(*b); 
	}

	prob[1]=exp(prob[1])/(1+exp(prob[1]));
	prob[0]=1-prob[1];
 
	// return class distribution
	return PDistribution(mlnew TDiscDistribution(prob, 2));
}


