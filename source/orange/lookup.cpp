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
#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "estimateprob.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "stladdon.hpp"
#include "distvars.hpp"
#include "bayes.hpp"
#include "contingency.hpp"
#include "table.hpp"

#include "lookup.ppp"

inline TValue getValue(const TExample &ex, const int &varIndex, PVariable variable)
{ return (varIndex==ILLEGAL_INT) ? variable->computeValue(ex) : ex[varIndex]; }


TClassifierByLookupTable::TClassifierByLookupTable(PVariable aclass, PValueList vlist)
: TClassifier(aclass, false), // we want TClassifier::classDistribution to call operator() when there are no distributions
  lookupTable(vlist),
  distributions(mlnew TDistributionList())
{ 
  for(int i = lookupTable->size(); i--; )
    distributions->push_back(TDistribution::create(aclass));
}


TClassifierByLookupTable1::TClassifierByLookupTable1(PVariable aclass, PVariable avar)
: TClassifierByLookupTable(aclass, mlnew TValueList(avar->noOfValues()+1, aclass->DK(), aclass)), 
  variable1(avar), 
  lastDomainVersion(-1)
{}


void TClassifierByLookupTable1::setLastDomain(PDomain domain)
{ lastVarIndex = domain->getVarNum(variable1);
  lastDomainVersion = domain->version;
}


int TClassifierByLookupTable1::getIndex(const TExample &ex, TExample *conv)
{ if (lastDomainVersion!=ex.domain->version) 
    setLastDomain(ex.domain);
  
  TValue val = getValue(ex, lastVarIndex, variable1);
  
  if (val.isSpecial()) {
    if (conv)
      (*conv)[0] = val;
    return -1;
  }
  
  return val.intV;
}


TValue TClassifierByLookupTable1::operator()(const TExample &ex)
{ if (lastDomainVersion!=ex.domain->version)
    setLastDomain(ex.domain);

  TValue val = getValue(ex, lastVarIndex, variable1);
  return (val.isSpecial() || (val.intV>=int(lookupTable->size())))
    ? lookupTable->back()
    : lookupTable->operator[](val.intV);
}


PDistribution TClassifierByLookupTable1::classDistribution(const TExample &ex)
{ if (!distributions)
    return TClassifier::classDistribution(ex);

  if (lastDomainVersion!=ex.domain->version)
    setLastDomain(ex.domain);

  TValue val = getValue(ex, lastVarIndex, variable1);
  return (val.isSpecial() || (val.intV>=int(distributions->size())))
    ? CLONE(TDistribution, distributions->back())
    : CLONE(TDistribution, distributions->operator[](val.intV));
}


void TClassifierByLookupTable1::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{ if (!distributions) {
    TClassifier::predictionAndDistribution(ex, value, dist);
    return;
  }

  if (lastDomainVersion!=ex.domain->version)
    setLastDomain(ex.domain);

  TValue val = getValue(ex, lastVarIndex, variable1);
  if (val.isSpecial() || (val.intV>=int(lookupTable->size()))) {
    value = lookupTable->back();
    dist = CLONE(TDistribution, distributions->back());
  }
  else {
    value = lookupTable->operator[](val.intV);
    dist = CLONE(TDistribution, distributions->operator[](val.intV));
  }
}


/* valDistribution is distribution of values for variable
   If "distributions" are known, this computes the weighted sum of distributions
   and stores it in lookup table so that they're used anywhere where the table element is missing
   Otherwise, it computes the distribution of classes and stores the majority into empty
   elements of the lookup table.
*/
void TClassifierByLookupTable1::replaceDKs(TDiscDistribution &valDistribution)
{ TDiscDistribution sum(classVar), *classes = mlnew TDiscDistribution(classVar);
  PDistribution wclasses = PDistribution(classes);
  if (distributions) {
    if (valDistribution.abs) {
      vector<TValue>::iterator vi(lookupTable->begin());
      valDistribution[distributions->size()-1];
      TDiscDistribution::iterator di(valDistribution.begin());
      for(vector<PDistribution>::iterator dvi(distributions->begin()), dve(distributions->end());
          dvi!=dve;
          dvi++, vi++, di++)
        if (!(*vi).isSpecial()) {
          const TDiscDistribution &tdi = CAST_TO_DISCDISTRIBUTION(*dvi);
          TDiscDistribution::const_iterator tdii(tdi.begin());
          for(TDiscDistribution::iterator si(sum.begin()), se(sum.end()); si!=se; si++)
            *si += *tdii * *di;
          sum.abs += tdi.abs * *di;
          classes->addint((*vi).intV, *di);
        }
    }
    else {
      vector<TValue>::iterator vi(lookupTable->begin());
      for(vector<PDistribution>::iterator dvi(distributions->begin()), dve(distributions->end()); dvi!=dve; dvi++, vi++)
        if (!(*vi).isSpecial()) {
          sum += CAST_TO_DISCDISTRIBUTION(*dvi);
          classes->addint((*vi).intV, 1);
        }
    }

    sum.normalize();
    vector<TValue>::iterator vi(lookupTable->begin());
    PITERATE(vector<PDistribution>, dvi, distributions)
      if ((*vi).isSpecial()) {
        *dvi = mlnew TDiscDistribution(sum);
        *(vi++) = classes->highestProbValue(); // this does not need to be the same for each call!
      }
      else
        vi++;
  }
  else {
    TDiscDistribution::iterator di(valDistribution.begin());
    if (valDistribution.abs) {
      for(vector<TValue>::iterator vi(lookupTable->begin()), ve(lookupTable->end()); vi!=ve; vi++, di++)
        if (!(*vi).isSpecial())
          classes->addint((*vi).intV, *di);
    }
    else {
      for(vector<TValue>::iterator vi(lookupTable->begin()), ve(lookupTable->end()); vi!=ve; vi++, di++)
        if (!(*vi).isSpecial())
          classes->addint((*vi).intV, 1.0);
    }

    PITERATE(vector<TValue>, vi, lookupTable)
      if ((*vi).isSpecial())
        (*vi) = classes->highestProbValue(); // this does not need to be the same for each call!
  }
}


void TClassifierByLookupTable1::giveBoundSet(TVarList &boundSet)
{ boundSet=TVarList(1, variable1); }




TClassifierByLookupTable2::TClassifierByLookupTable2(PVariable aclass, PVariable avar1, PVariable avar2, PEFMDataDescription adata)
: TClassifierByLookupTable(aclass, mlnew TValueList((avar1->noOfValues()) * (avar2->noOfValues()), aclass->DK(), aclass)), 
  variable1(avar1),
  variable2(avar2),
  noOfValues1(avar1->noOfValues()),
  noOfValues2(avar2->noOfValues()),
  dataDescription(adata),
  lastDomainVersion(-1)
{ if (!adata) {
    TVarList attributes;
    attributes.push_back(variable1);
    attributes.push_back(variable2);
    dataDescription=mlnew TEFMDataDescription(mlnew TDomain(PVariable(), attributes)); 
  }
}


void TClassifierByLookupTable2::setLastDomain(PDomain domain)
{ lastVarIndex1 = domain->getVarNum(variable1);
  lastVarIndex2 = domain->getVarNum(variable2);
  lastDomainVersion = domain->version;
}


int TClassifierByLookupTable2::getIndex(const TExample &ex, TExample *conv)
{ if (lastDomainVersion!=ex.domain->version) 
    setLastDomain(ex.domain);
  
  TValue val1 = getValue(ex, lastVarIndex1, variable1);
  TValue val2 = getValue(ex, lastVarIndex2, variable2);
  
  if (val1.isSpecial() || val2.isSpecial()) {
    if (conv) {
      (*conv)[0] = val1;
      (*conv)[1] = val2;
    }
    return -1;
  }
  
  return noOfValues2 * val1.intV + val2.intV;
}


TValue TClassifierByLookupTable2::operator()(const TExample &ex)
{ TExample conv(dataDescription->domain);
  
  int index=getIndex(ex, &conv);
  if (index<0)
    return TClassifier::operator()(conv, dataDescription);
  else if (index>=int(lookupTable->size()))
    return dataDescription->domainDistributions->back()->highestProbValue(ex);
  else 
    return lookupTable->operator[](index);
}


PDistribution TClassifierByLookupTable2::classDistribution(const TExample &ex)
{ if (!distributions)
    return TClassifier::classDistribution(ex);

  TExample conv(dataDescription->domain);

  int index=getIndex(ex, &conv);
  if (index<0) 
    return TClassifier::classDistribution(conv, dataDescription);
  else if (index>=int(distributions->size()))
    return CLONE(TDistribution, dataDescription->domainDistributions->back());
  else
    return CLONE(TDistribution, distributions->operator[](index));
}


void TClassifierByLookupTable2::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{ if (!distributions) {
    TClassifier::predictionAndDistribution(ex, value, dist);
    return;
  }

  TExample conv(dataDescription->domain);

  int index=getIndex(ex, &conv);
  if (index<0) {
    dist = TClassifier::classDistribution(conv, dataDescription);
    value = dist->highestProbValue(ex);
  }
  else if (index>=int(distributions->size())) {
    dist = CLONE(TDistribution, dataDescription->domainDistributions->back());
    value = dist->highestProbValue(ex);
  }
  else {
    dist = CLONE(TDistribution, distributions->operator[](index));
    value = lookupTable->operator[](index);
  }
}


void TClassifierByLookupTable2::replaceDKs(PExampleGenerator examples, bool useBayes)
{
  PClassifier bayes;
  PDistribution classDist;

  if (useBayes)
    bayes = TBayesLearner()(examples);
  else
    classDist =  getClassDistribution(examples /*, weightID */);

  vector<TValue>::iterator vi(lookupTable->begin());
  vector<PDistribution>::iterator di(distributions->begin());
  bool distr = distributions && (distributions->size()>0);
  TExample example(dataDescription->domain);
  variable1->firstValue(example[0]);
  do {
    variable2->firstValue(example[1]);
    do {
      if ((*vi).isSpecial()) 
        if (useBayes) {
          *vi = bayes->operator()(example);
          if (distr)
            *di = bayes->classDistribution(example);
        }
        else {
          *vi = classDist->highestProbValue(example);
          if (distr) 
            *di = CLONE(TDistribution, classDist);
        }
      vi++;
      if (distr) 
        di++;
    } while(variable2->nextValue(example[1]));
  } while (variable1->nextValue(example[0]));
}


void TClassifierByLookupTable2::giveBoundSet(TVarList &boundSet)
{ boundSet=TVarList();
  boundSet.push_back(variable1);
  boundSet.push_back(variable2);
}



TClassifierByLookupTable3::TClassifierByLookupTable3(PVariable aclass, PVariable avar1, PVariable avar2, PVariable avar3, PEFMDataDescription adata)
: TClassifierByLookupTable(aclass, mlnew TValueList((avar1->noOfValues()) * (avar2->noOfValues()) * (avar3->noOfValues()), aclass->DK(), aclass)),
  variable1(avar1),
  variable2(avar2),
  variable3(avar3),
  noOfValues1(avar1->noOfValues()),
  noOfValues2(avar2->noOfValues()),
  noOfValues3(avar3->noOfValues()),
  dataDescription(adata),
  lastDomainVersion(-1)
{ if (!adata) {
    TVarList attributes;
    attributes.push_back(variable1);
    attributes.push_back(variable2);
    attributes.push_back(variable3);
    dataDescription = mlnew TEFMDataDescription(mlnew TDomain(PVariable(), attributes)); 
  }
}


void TClassifierByLookupTable3::setLastDomain(PDomain domain)
{ 
  lastVarIndex1 = domain->getVarNum(variable1);
  lastVarIndex2 = domain->getVarNum(variable2);
  lastVarIndex3 = domain->getVarNum(variable3);
  lastDomainVersion=domain->version;
}


int TClassifierByLookupTable3::getIndex(const TExample &ex, TExample *conv)
{  if (lastDomainVersion!=ex.domain->version)
     setLastDomain(ex.domain);
  
  TValue val1=getValue(ex, lastVarIndex1, variable1);
  TValue val2=getValue(ex, lastVarIndex2, variable2);
  TValue val3=getValue(ex, lastVarIndex3, variable3);

   if (val1.isSpecial() || val2.isSpecial() || val3.isSpecial()) {
     if (conv) {
       (*conv)[0]=val1;
       (*conv)[1]=val2;
       (*conv)[2]=val3;
     }
    return -1;
  }
  
  return noOfValues3 * (noOfValues2 * val1.intV + val2.intV) + val3.intV;
}


TValue TClassifierByLookupTable3::operator()(const TExample &ex)
{ TExample conv(dataDescription->domain);
  
  int index=getIndex(ex, &conv);
  if (index<0)
    return TClassifier::operator()(conv, dataDescription);
  else if (index>=int(lookupTable->size()))
    return dataDescription->domainDistributions->back()->highestProbValue(ex);
  else 
    return lookupTable->operator[](index);
}


PDistribution TClassifierByLookupTable3::classDistribution(const TExample &ex)
{ if (!distributions)
    return TClassifier::classDistribution(ex);

  TExample conv(dataDescription->domain);

  int index=getIndex(ex, &conv);
  if (index<0) 
    return TClassifier::classDistribution(conv, dataDescription);
  else if (index>=int(distributions->size()))
    return CLONE(TDistribution, dataDescription->domainDistributions->back());
  else
    return CLONE(TDistribution, distributions->operator[](index));
}


void TClassifierByLookupTable3::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{ if (!distributions) {
    TClassifier::predictionAndDistribution(ex, value, dist);
    return;
  }

  TExample conv(dataDescription->domain);

  int index=getIndex(ex, &conv);
  if (index<0) {
    dist = TClassifier::classDistribution(conv, dataDescription);
    value = dist->highestProbValue(ex);
  }
  else if (index>=int(distributions->size())) {
    dist = CLONE(TDistribution, dataDescription->domainDistributions->back());
    value = dist->highestProbValue(ex);
  }
  else {
    dist = CLONE(TDistribution, distributions->operator[](index));
    value = lookupTable->operator[](index);
  }
}


void TClassifierByLookupTable3::replaceDKs(PExampleGenerator examples, bool useBayes)
{
  PClassifier bayes;
  PDistribution classDist;

  if (useBayes)
    bayes = TBayesLearner()(examples);
  else
    classDist = getClassDistribution(examples /*, weight */);

  vector<TValue>::iterator vi(lookupTable->begin());
  vector<PDistribution>::iterator di(distributions->begin());
  bool distr=distributions && (distributions->size()>0);
  TExample example(dataDescription->domain);
  variable1->firstValue(example[0]);
  do {
    variable2->firstValue(example[1]);
    do {
      variable3->firstValue(example[2]);
      do {
        if ((*vi).isSpecial()) 
          if (useBayes) {
            *vi=bayes->operator()(example);
            if (distr) *di=bayes->classDistribution(example);
          }
          else {
            *vi = classDist->highestProbValue(example);
            if (distr) 
              *di = CLONE(TDistribution, classDist);
          }
        vi++;
        if (distr)
          di++;
      } while (variable3->nextValue(example[2]));
    } while (variable2->nextValue(example[1]));
  } while (variable1->nextValue(example[0]));
}

void TClassifierByLookupTable3::giveBoundSet(TVarList &boundSet)
{ boundSet = TVarList();
  boundSet.push_back(variable1);
  boundSet.push_back(variable2);
  boundSet.push_back(variable3);
}






TLookupLearner::TLookupLearner()
{}

TLookupLearner::TLookupLearner(const TLookupLearner &old)
: TLearner(old)
{}


PClassifier TLookupLearner::operator()(PExampleGenerator ogen, const int &weight)
{ if (!ogen->domain->classVar)
    raiseError("class-less domain");

  PExampleGenerator gen = fixedExamples(ogen);
  TExampleTable examplePtrs(gen, false);
  examplePtrs.sort();

  TClassifierByExampleTable *classifier = mlnew TClassifierByExampleTable(examplePtrs.domain);
  PClassifier wclassifier = PClassifier(classifier);
  
  int attrs = examplePtrs.domain->attributes->size();

  TExampleIterator bi(examplePtrs.begin());
  while (bi) {
    TExampleIterator bbi = bi;
    PDistribution classDist = TDistribution::create(examplePtrs.domain->classVar);
    TDistribution &tcv = classDist.getReference();
    int diff;
    do {
      if (!(*bbi).getClass().isSpecial())
        tcv.add((*bbi).getClass(), WEIGHT2(*bbi, weight));
      if (!++bbi)
        break;
      TExample::iterator bii((*bi).begin()), bbii((*bbi).begin());
      for(diff = attrs; diff && (*(bii++)==*(bbii++)); diff--);
    } while (!diff);

    if (classDist->abs > 0.0) {
      TExample ex = *bi;
      ex.setClass(tcv.highestProbValue(ex));
      ex.getClass().svalV = classDist;

      classifier->sortedExamples->addExample(ex);
    }
    bi = bbi;
  }

  if (learnerForUnknown)
    classifier->classifierForUnknown = learnerForUnknown->operator()(ogen, weight);

  return wclassifier;
}



TClassifierByExampleTable::TClassifierByExampleTable(PDomain dom)
: TClassifierFD(dom),
  domainWithoutClass(CLONE(TDomain, dom)),
  sortedExamples(mlnew TExampleTable(dom))
{ domainWithoutClass->removeClass(); }


TClassifierByExampleTable::TClassifierByExampleTable(PExampleGenerator gen, PClassifier unk)
: TClassifierFD(gen->domain),
  domainWithoutClass(CLONE(TDomain, gen->domain)),
  sortedExamples(mlnew TExampleTable(gen)),
  classifierForUnknown(unk)
{ domainWithoutClass->removeClass(); }


void TClassifierByExampleTable::getExampleRange(const TExample &exam, TExample **&low, TExample **&high)
{
  /* There's a reason why we need domainWithoutClass.
     TClassifierByExampleTable is often used for getValueFrom. If it
     tried to convert the example to a domain with the class,
     this would also convert the class attribute, which would
     usually trigger a call to its getValueFrom and so forth
     until a stack overflow :) */

  TExample convertedEx(domainWithoutClass, exam);
  low = sortedExamples->examples;
  high = sortedExamples->_Last;
  TExample **tee;
  for(int aind=0, aend=domain->attributes->size(); (aind<aend) && (low!=high); aind++) {
    while( (low!=high) && ((**low)[aind].compare(convertedEx[aind])<0))
      low++;
    if (low!=high) {
      tee = low;
      while ( (tee!=high) && ((**tee)[aind]==(**low)[aind]))
        tee++;
      high = tee;
    }
  }

  if (low == high)
    low = high = NULL;
}


PDistribution TClassifierByExampleTable::classDistributionLow(TExample **low, TExample **high)
{
  PDistribution res;
  for(; low!=high; low++) {
    TDistribution *ures = NULL;

    TValue cval = (**low).getClass();
    if (!cval.svalV || !cval.svalV.is_derived_from(TDistribution))
      raiseError("invalid value type");

    if (!ures) {
      ures = CLONE(TDistribution, cval.svalV);
      res = ures;
    }
    else
      (*ures) += cval.svalV;
  }

  return res;
}


TValue TClassifierByExampleTable::operator()(const TExample &exam)
{ TExample **low, **high;
  getExampleRange(exam, low, high);

  if (low && (low==high-1))
    return (*low)->getClass();

  PDistribution probs = classDistributionLow(low, high);

  if (probs)
    //  might be that low was NULL or classDistributionLow returned NULL
    return probs->highestProbValue(exam);
  else
    return classifierForUnknown ? classifierForUnknown->operator()(exam) : domain->classVar->DK();
}


PDistribution TClassifierByExampleTable::classDistribution(const  TExample &exam)
{ TExample **low, **high;
  getExampleRange(exam, low, high);

  PDistribution dval = classDistributionLow(low, high);
  if (dval) {
    PDistribution dd = CLONE(TDistribution, dval);
    dval->normalize();
    return dval;
  }
 
  if (classifierForUnknown)
    return classifierForUnknown->classDistribution(exam);

  dval = TDistribution::create(domain->classVar);
  dval->normalize();
  return PDistribution();
}


void TClassifierByExampleTable::predictionAndDistribution(const TExample &exam, TValue &pred, PDistribution &dist)
{ TExample **low, **high;
  getExampleRange(exam, low, high);

  PDistribution dval = classDistributionLow(low, high);
  if (dval) {
    pred = dval->highestProbValue(exam);
    dist = CLONE(TDistribution, dval);
    dist->normalize();
  }

  else if (classifierForUnknown)
    classifierForUnknown->predictionAndDistribution(exam, pred, dist);

  else {
    pred = domain->classVar->DK();
    dval = TDistribution::create(domain->classVar);
    dval->normalize();
  }
}


void TClassifierByExampleTable::afterSet(const char *name)
{
  if (!strcmp(name, "sortedExamples")) {
    domain = sortedExamples->domain; 
    classVar = sortedExamples->domain->classVar;
    domainWithoutClass = CLONE(TDomain, domain);
    domainWithoutClass->removeClass();
  }

  TClassifierFD::afterSet(name);
}




TClassifierFromGenerator::TClassifierFromGenerator() 
: weightID(0)
{ computesProbabilities=true; }


TClassifierFromGenerator::TClassifierFromGenerator(PVariable &acv)
: TDefaultClassifier(acv),
  weightID(0),
  dataDescription()
{ computesProbabilities = true; }


TClassifierFromGenerator::TClassifierFromGenerator(PVariable &acv, TValue &val, TDistribution &dval)
: TDefaultClassifier(acv, val, CLONE(TDistribution, &dval)),
  weightID(0),
  dataDescription()
{ computesProbabilities = true; }


TClassifierFromGenerator::TClassifierFromGenerator(PExampleGenerator agen, int aWeightID)
: TDefaultClassifier(agen->domain->classVar),
  generator(mlnew TExampleTable(agen)),
  weightID(aWeightID),
  domainWithoutClass(CLONE(TDomain, agen->domain)),
  dataDescription(mlnew TEFMDataDescription(agen->domain, mlnew TDomainDistributions(agen), aWeightID, getMetaID()))
{ domainWithoutClass->removeClass(); 
  computesProbabilities=true; }


TClassifierFromGenerator::TClassifierFromGenerator(const TClassifierFromGenerator &old)
: TDefaultClassifier(old), generator(old.generator),
  weightID(old.weightID),
  domainWithoutClass(old.domainWithoutClass),
  dataDescription(old.dataDescription)
{}


#include "filter.hpp"


TValue TClassifierFromGenerator::operator ()(const TExample &exam)
{ static TFilter_hasSpecial hasSpecial;

  TExample cexam(domainWithoutClass, exam);

  if (hasSpecial(exam)) {
    TExample exam2(dataDescription->domain, cexam);
    return TClassifier::operator()(exam2, dataDescription);
  }

  PDistribution wclassDist = TDistribution::create(generator->domain->classVar);
  TDistribution &classDist = wclassDist.getReference();
  for(TExampleIterator ri(generator->begin()); ri; ++ri)
    if (cexam.compatible(*ri) && !(*ri).getClass().isSpecial())
      classDist.add((*ri).getClass(), WEIGHT(*ri));
    
  if (classDist.abs)
    return classDist.highestProbValue(exam);
  else
    return classifierForUnknown ? classifierForUnknown->operator()(exam) : classVar->DK();
}


PDistribution TClassifierFromGenerator::classDistribution(const TExample &exam)
{ TExample cexam(domainWithoutClass, exam);

  if (TFilter_hasSpecial()(exam)) {
    TExample exam2(dataDescription->domain, cexam);
    return TClassifier::classDistribution(exam, dataDescription);
  }
  
  PDistribution wclassDist = TDistribution::create(generator->domain->classVar);
  TDistribution &classDist = wclassDist.getReference();
  for(TExampleIterator ri(generator->begin()); ri; ++ri)
    if (cexam.compatible(*ri) && !(*ri).getClass().isSpecial())
      classDist.add((*ri).getClass(), WEIGHT(*ri));

  return wclassDist;
}
