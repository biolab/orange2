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
#include "table.hpp"
#include "logistic.ppp"
#include <math.h>


TLogRegLearner::TLogRegLearner() 
{}

// TODO: najdi pametno mesto za naslednji dve funkciji
// compute waldZ statistic from beta and beta_se
PAttributedFloatList TLogRegLearner::computeWaldZ(PAttributedFloatList &beta, PAttributedFloatList &beta_se) 
{
  PAttributedFloatList waldZ=PAttributedFloatList(mlnew TAttributedFloatList(beta->attributes));
  TAttributedFloatList::const_iterator b(beta->begin()), be(beta->end());
  TAttributedFloatList::const_iterator s(beta_se->begin()), se(beta_se->end());
  for (; (b!=be) && (s!=se); b++, s++) 
    waldZ->push_back((*b)/(*s));
   return waldZ;
}

// compute P from waldZ statistic
PAttributedFloatList TLogRegLearner::computeP(PAttributedFloatList &waldZ) 
{
  PAttributedFloatList Pstat=PAttributedFloatList(mlnew TAttributedFloatList(waldZ->attributes));
  TAttributedFloatList::const_iterator z(waldZ->begin()), ze(waldZ->end());
  for (; (z!=ze); z++) {
    double zt = (*z)*(*z);
    if(zt>1000) {
      Pstat->push_back(0.0);
      continue;
    }
    double p = exp(-0.5*zt);
    // TODO: PI, kje najdes to konstano
    p *= sqrt(2*zt/3.141592);

    double t=p;
    int a=3;
    // TODO: poglej kaj je to 0.0000...1 ?
    for (; t>0.0000000001*p; a=a+2) {
      t*=zt/a; 
      p+=t;
    }
    Pstat->push_back(1-p);
    }
	return Pstat;
}


PClassifier TLogRegLearner::operator()(PExampleGenerator gen, const int &weight)
{ 
  int error;
  PVariable var;
  PClassifier cl = fitModel(gen, weight, error, var);

  if (error >= TLogRegFitter::Constant)
    raiseError("%s in %s", error==TLogRegFitter::Constant ? "constant" : "singularity", var->name.c_str());

  return cl;
}


TDomainContinuizer *constructDefaultLRContinuizer()
{ 
  TDomainContinuizer *def = mlnew TDomainContinuizer();
  def->zeroBased = true;
  def->continuousTreatment = TDomainContinuizer::Leave;
  def->multinomialTreatment = TDomainContinuizer::FrequentIsBase;
  def->classTreatment = TDomainContinuizer::Ignore;
  return def;
}


TDomainContinuizer *logisticRegressionDomainContinuizer = constructDefaultLRContinuizer();

PClassifier TLogRegLearner::fitModel(PExampleGenerator gen, const int &weight, int &error, PVariable &errorAt)
{ 
  PImputer imputer = imputerConstructor ? imputerConstructor->call(gen, weight) : PImputer();
  PExampleGenerator imputed = imputer ? imputer->call(gen, weight) : gen;

  // construct classifier	
  TLogRegClassifier *lrc = mlnew TLogRegClassifier(imputed->domain);
  lrc->dataDescription = mlnew TEFMDataDescription(gen->domain, mlnew TDomainDistributions(gen), 0, getMetaID());
  PClassifier cl = lrc;
  lrc->imputer = imputer;

  //if (imputed->domain->hasDiscreteAttributes(false)) {
    lrc->continuizedDomain = domainContinuizer ? domainContinuizer->call(imputed, weight) : (*logisticRegressionDomainContinuizer)(imputed, weight);
    imputed = mlnew TExampleTable(lrc->continuizedDomain, imputed);
  //}

    // copy class value

  // construct a LR fitter
  fitter = fitter ? fitter : PLogRegFitter(mlnew TLogRegFitter_Cholesky());

  PAttributedFloatList temp_beta, temp_beta_se;
  // fit logistic regression 

  temp_beta = fitter->call(imputed, weight, temp_beta_se, lrc->likelihood, error, errorAt);
  lrc->fit_status = error;

  // transform beta to AttributedList
  PVarList enum_attributes = mlnew TVarList(); 
  enum_attributes->push_back(imputed->domain->classVar);
  PITERATE(TVarList, vl, imputed->domain->attributes) 
    enum_attributes->push_back(*vl);
  // tranfsorm *beta into a PFloatList
  lrc->beta=mlnew TAttributedFloatList(enum_attributes);
  lrc->beta_se=mlnew TAttributedFloatList(enum_attributes);

  PITERATE(TAttributedFloatList, fi, temp_beta)
    lrc->beta->push_back(*fi);

  PITERATE(TAttributedFloatList, fi_se, temp_beta_se)
    lrc->beta_se->push_back(*fi_se);

  if (error >= TLogRegFitter::Constant) 
    return cl;

  lrc->wald_Z = computeWaldZ(lrc->beta, lrc->beta_se);
  lrc->P = computeP(lrc->wald_Z);

  // return classifier with domain, beta and standard errors of beta 
  return cl;
}


TLogRegClassifier::TLogRegClassifier() 
{}


TLogRegClassifier::TLogRegClassifier(PDomain dom) 
: TClassifierFD(dom, true)
{};


PDistribution TLogRegClassifier::classDistribution(const TExample &origexam)
{   
  checkProperty(domain);
  TExample cexample(domain, origexam);

  TExample *example2;

  if (imputer)
    example2 = imputer->call(cexample);
  else {
    if (dataDescription)
      for(TExample::const_iterator ei(cexample.begin()), ee(cexample.end()-1); ei!=ee; ei++)
        if ((*ei).isSpecial())
          return TClassifier::classDistribution(cexample, dataDescription);

    example2 = &cexample;
  }

  TExample *example = continuizedDomain ? mlnew TExample(continuizedDomain, *example2) : example2;

  float prob1;
  try {
    // multiply example with beta
    TAttributedFloatList::const_iterator b(beta->begin()), be(beta->end());

    // get beta 0
    prob1 = *b;
    b++;
    // multiply beta with example
    TVarList::const_iterator vi(example->domain->attributes->begin());
    TExample::const_iterator ei(example->begin()), ee(example->end());
    for (; (b!=be) && (ei!=ee); ei++, b++, vi++) {
      if ((*ei).isSpecial())
        raiseError("unknown value in attribute '%s'", (*vi)->name.c_str());
      prob1 += (*ei).floatV * (*b); 
    }

    prob1 = exp(prob1)/(1+exp(prob1));
  }
  catch (...) {
    if (imputer)
      mldelete example2;
    if (continuizedDomain)
      mldelete example;
    throw;
  }

  if (imputer)
    mldelete example2;
  if (continuizedDomain)
    mldelete example;

  if (classVar->varType == TValue::INTVAR) {
      TDiscDistribution *dist = mlnew TDiscDistribution(classVar);
      PDistribution res = dist;
      dist->addint(0, 1-prob1);
      dist->addint(1, prob1);
      return res;
  }
  else {
      TContDistribution *dist = mlnew TContDistribution(classVar);
      PDistribution res = dist;
      dist->addfloat(prob1, 1.0);
      return res;
  }
}
