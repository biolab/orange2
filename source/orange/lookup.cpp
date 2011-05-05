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
#include "filter.hpp"

#include "lookup.ppp"

inline TValue getValue(const TExample &ex, const int &varIndex, PVariable variable)
{ return (varIndex==ILLEGAL_INT) ? variable->computeValue(ex) : ex[varIndex]; }



TClassifierByLookupTable::TClassifierByLookupTable(PVariable aclass, PValueList vlist)
: TClassifier(aclass, false), // we want TClassifier::classDistribution to call operator() when there are no distributions
  lookupTable(vlist),
  distributions(mlnew TDistributionList())
{ 
  if (lookupTable)
    for(int i = lookupTable->size(); i--; )
      distributions->push_back(TDistribution::create(aclass));
}


void TClassifierByLookupTable::valuesFromDistributions()
{
  if (lookupTable->size() != distributions->size())
    raiseError("sizes of 'lookupTable' and 'distributions' mismatch");

  TValueList::iterator vi(lookupTable->begin());
  TDistributionList::const_iterator di(distributions->begin()), de(distributions->end());
  for(; di!=de; di++, vi++)
    if ((*vi).isSpecial())
      *vi = (*di)->highestProbValue();
}


TClassifierByLookupTable1::TClassifierByLookupTable1(PVariable aclass, PVariable avar)
: TClassifierByLookupTable(aclass, mlnew TValueList(avar->noOfValues()+1, aclass->DK(), aclass)), 
  variable1(avar), 
  lastDomainVersion(-1)
{}


void TClassifierByLookupTable1::setLastDomain(PDomain domain)
{ lastVarIndex = domain->getVarNum(variable1, false);
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
      TValueList::iterator vi(lookupTable->begin());
      //valDistribution[distributions->size()-1];
      TDiscDistribution::iterator di(valDistribution.begin()), de(valDistribution.end());
      for(TDistributionList::iterator dvi(distributions->begin()), dve(distributions->end());
          (dvi!=dve) && (di!=de);
          dvi++, vi++, di++)
        if (!(*vi).isSpecial()) {
          const TDiscDistribution &tdi = CAST_TO_DISCDISTRIBUTION(*dvi);
          TDiscDistribution::const_iterator tdii(tdi.begin());
          for(TDiscDistribution::iterator si(sum.begin()), se(sum.end()); si!=se; si++, tdii++)
            *si += *tdii * *di;
          sum.abs += tdi.abs * *di;
          classes->addint((*vi).intV, *di);
        }
    }
    else {
      TValueList::iterator vi(lookupTable->begin());
      for(TDistributionList::iterator dvi(distributions->begin()), dve(distributions->end()); dvi!=dve; dvi++, vi++)
        if (!(*vi).isSpecial()) {
          sum += CAST_TO_DISCDISTRIBUTION(*dvi);
          classes->addint((*vi).intV, 1);
        }
    }

    sum.normalize();
    TValueList::iterator vi(lookupTable->begin());
    PITERATE(TDistributionList, dvi, distributions)
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
      for(TValueList::iterator vi(lookupTable->begin()), ve(lookupTable->end()); vi!=ve; vi++, di++)
        if (!(*vi).isSpecial())
          classes->addint((*vi).intV, *di);
    }
    else {
      for(TValueList::iterator vi(lookupTable->begin()), ve(lookupTable->end()); vi!=ve; vi++, di++)
        if (!(*vi).isSpecial())
          classes->addint((*vi).intV, 1.0);
    }

    PITERATE(TValueList, vi, lookupTable)
      if ((*vi).isSpecial())
        (*vi) = classes->highestProbValue(); // this does not need to be the same for each call!
  }
}


void TClassifierByLookupTable1::giveBoundSet(TVarList &boundSet)
{ boundSet = TVarList(1, variable1); }




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
{ lastVarIndex1 = domain->getVarNum(variable1, false);
  lastVarIndex2 = domain->getVarNum(variable2, false);
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

  TValueList::iterator vi(lookupTable->begin());
  TDistributionList::iterator di(distributions->begin());
  bool distr = distributions && (distributions->size()>0);
  TExample example(dataDescription->domain);
  variable1->firstValue(example[0]);
  do {
    variable2->firstValue(example[1]);
    do {
      if ((*vi).isSpecial()) 
        if (useBayes) {
          if (distr) {
            *di = bayes->classDistribution(example);
            *vi = (*di)->highestProbValue(example);
          }
          else
            *vi = bayes->operator()(example);
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
  lastVarIndex1 = domain->getVarNum(variable1, false);
  lastVarIndex2 = domain->getVarNum(variable2, false);
  lastVarIndex3 = domain->getVarNum(variable3, false);
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

  TValueList::iterator vi(lookupTable->begin());
  TDistributionList::iterator di(distributions->begin());
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



TClassifierByLookupTableN::TClassifierByLookupTableN(PVariable aclass, PVarList avars, PEFMDataDescription adata)
: TClassifierByLookupTable(aclass, NULL),
  variables(avars),
  noOfValues(mlnew TIntList()),
  dataDescription(adata),
  lastDomainVersion(-1)
{ 
  long int totvals = 1;
  const_PITERATE(TVarList, ai, avars) {
    if ((*ai)->varType != TValue::INTVAR)
      raiseError("lookup tables only work with discrete attributes");
    noOfValues->push_back((*ai)->noOfValues());
    totvals *= (*ai)->noOfValues();
  }

  lookupTable = mlnew TValueList(totvals, aclass->DK(), aclass);

  distributions = mlnew TDistributionList();
  for(int i = totvals; i--; )
    distributions->push_back(TDistribution::create(aclass));

  if (!adata)
    dataDescription = mlnew TEFMDataDescription(mlnew TDomain(PVariable(), avars.getReference())); 
}


void TClassifierByLookupTableN::setLastDomain(PDomain domain)
{ 
  lastVarIndices.clear();
  const_PITERATE(TVarList, vi, variables)
    lastVarIndices.push_back(domain->getVarNum(*vi, false));
  lastDomainVersion = domain->version;
}


int TClassifierByLookupTableN::getIndex(const TExample &ex, TExample *conv)
{
  if (lastDomainVersion!=ex.domain->version)
     setLastDomain(ex.domain);

  int index = 0;
  TVarList::const_iterator vi(variables->begin());
  int i = 0;
  vector<int>::const_iterator ii(lastVarIndices.begin()), iie(lastVarIndices.end());
  TIntList::const_iterator ni(noOfValues->begin());
  for(; ii != iie; ii++, vi++, i++, ni++) {

    const TValue val = getValue(ex, *ii, *vi);

    if (val.isSpecial()) {
      if (conv)
        for(; ii != iie; (*conv)[i++] = getValue(ex, *ii++, *vi++));
      return -1;
    }

    index = index * *ni + val.intV;

    if (conv)
      (*conv)[i] = val;
  }

  return index;
}


TValue TClassifierByLookupTableN::operator()(const TExample &ex)
{ 
  TExample conv(dataDescription->domain);
  
  int index = getIndex(ex, &conv);
  if (index<0)
    return TClassifier::operator()(conv, dataDescription);
  else if (index >= int(lookupTable->size()))
    return dataDescription->domainDistributions->back()->highestProbValue(ex);
  else 
    return lookupTable->operator[](index);
}


PDistribution TClassifierByLookupTableN::classDistribution(const TExample &ex)
{
  if (!distributions)
    return TClassifier::classDistribution(ex);

  TExample conv(dataDescription->domain);

  int index = getIndex(ex, &conv);
  if (index < 0) 
    return TClassifier::classDistribution(conv, dataDescription);
  else if (index >= int(distributions->size()))
    return CLONE(TDistribution, dataDescription->domainDistributions->back());
  else
    return CLONE(TDistribution, distributions->operator[](index));
}


void TClassifierByLookupTableN::predictionAndDistribution(const TExample &ex, TValue &value, PDistribution &dist)
{ 
  if (!distributions) {
    TClassifier::predictionAndDistribution(ex, value, dist);
    return;
  }

  TExample conv(dataDescription->domain);

  int index = getIndex(ex, &conv);
  if (index < 0) {
    dist = TClassifier::classDistribution(conv, dataDescription);
    value = dist->highestProbValue(ex);
  }
  else if (index >= int(distributions->size())) {
    dist = CLONE(TDistribution, dataDescription->domainDistributions->back());
    value = dist->highestProbValue(ex);
  }
  else {
    dist = CLONE(TDistribution, distributions->operator[](index));
    value = lookupTable->operator[](index);
  }
}


void TClassifierByLookupTableN::replaceDKs(PExampleGenerator examples, bool useBayes)
{
  raiseWarning("ClassifierByLookupTableN does not provide the function for replacing undefined values yet");
}

void TClassifierByLookupTableN::giveBoundSet(TVarList &boundSet)
{ 
  boundSet = variables.getReference();
}








TLookupLearner::TLookupLearner()
: unknownsHandling(UnknownsKeep),
  allowFastLookups(false)
{}



#define UNKNOWN_CLASS_WARNING \
{ \
  if (!alreadyWarned) { \
    raiseWarning("examples with unknown class are ignored"); \
    alreadyWarned = true; \
  } \
}


PClassifier TLookupLearner::operator()(PExampleGenerator ogen, const int &weightID)
{ 
  if (!ogen->domain->classVar)
    raiseError("class-less domain");

  const TVarList &attributes = ogen->domain->attributes.getReference();
  const int nattrs = attributes.size();
  PVariable classVar = ogen->domain->classVar;

  bool alreadyWarned = false;

  // we shall use ClassifierByLookupTable if the number of attributes
  // is <= 3 and the are all discrete
  if (allowFastLookups && (nattrs <= 3)) {
    TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
    for(; (vi!=ve) && ((*vi)->varType == TValue::INTVAR); vi++);
    if (vi==ve) {

      if (!nattrs) {
        PDistribution classDist = getClassDistribution(ogen, weightID);
        return mlnew TDefaultClassifier(classVar, classDist->highestProbValue(), classDist);
      }
    
      else if (nattrs == 1) {
        TClassifierByLookupTable1 *cblt = mlnew TClassifierByLookupTable1(classVar, attributes[0]);
        PClassifier wcblt = cblt;

        TDiscDistribution valDist(attributes[0]);
        TDiscDistribution unkDist(attributes[0]);

        PEITERATE(ei, ogen) {
          if ((*ei).getClass().isSpecial())
            UNKNOWN_CLASS_WARNING
          else {
            const TValue val = (*ei)[0];
            const float weight = WEIGHT(*ei);

            if (val.isSpecial()) {
              if (unknownsHandling)
                unkDist.addint((*ei)[1], weight);
            }
            else {
              cblt->distributions->at(val.intV)->addint((*ei)[1], weight);
              valDist.addint(val.intV, weight);
            }
          }
        }

        if (unkDist.abs && valDist.abs) {
          TDistributionList::iterator dli(cblt->distributions->begin());
          TDiscDistribution::const_iterator vdi(valDist.begin()), vde(valDist.end());
          for(; vdi!=vde; (dynamic_cast<TDiscDistribution &>((*dli++).getReference())).adddist(unkDist, *vdi++));
        }

        cblt->replaceDKs(valDist);
        cblt->valuesFromDistributions();
        return wcblt;
      }

      else {
        TClassifierByLookupTable *cblt = 
          nattrs == 2 ? (TClassifierByLookupTable *)mlnew TClassifierByLookupTable2(classVar, attributes[0], attributes[1])
                      : (TClassifierByLookupTable *)mlnew TClassifierByLookupTable3(classVar, attributes[0], attributes[1], attributes[2]);

        PClassifier wcblt = cblt;

        TExampleIterator ei(ogen->begin());
        for(; ei; ++ei) {
          if ((*ei).getClass().isSpecial())
            UNKNOWN_CLASS_WARNING
          else {
            const int idx = cblt->getIndex(*ei);
            if (idx<0) {
              raiseWarning("unknown attribute values detected: constructing ClassifierByExampleTable instead of LookupClassifier");
              break;
            }
            cblt->distributions->at(idx)->addint((*ei)[nattrs].intV, WEIGHT(*ei));
          }
        }

        if (!ei) { // have we finished prematurely due to unknown values?
          if (nattrs==2)
            dynamic_cast<TClassifierByLookupTable2 *>(cblt)->replaceDKs(ogen);
          else
            dynamic_cast<TClassifierByLookupTable3 *>(cblt)->replaceDKs(ogen);

          cblt->valuesFromDistributions();
          return wcblt;
        }
        // else fallthrough
      }
    }
  }


  PExampleGenerator gen = fixedExamples(ogen);
  TExampleTable examplePtrs(gen, false);
  examplePtrs.sort();

  TExampleTable unknowns(gen->domain);

  TEFMDataDescription *efmdata = mlnew TEFMDataDescription(gen->domain, mlnew TDomainDistributions(gen), weightID, getMetaID());
  PEFMDataDescription wefmdata = efmdata;

  TClassifierByExampleTable *classifier = mlnew TClassifierByExampleTable(examplePtrs.domain);
  PClassifier wclassifier = PClassifier(classifier);
  classifier->dataDescription = wefmdata;  

  TFilter_hasSpecial hasSpecial;
  
  for (TExampleIterator bi(examplePtrs.begin()), bbi(bi); bi; bi = bbi) {
    PDistribution classDist = TDistribution::create(examplePtrs.domain->classVar);
    TDistribution &tcv = classDist.getReference();

    if ((*bbi).getClass().isSpecial()) {
      UNKNOWN_CLASS_WARNING
      continue;
    }

    int diff;
    do {
      tcv.add((*bbi).getClass(), WEIGHT2(*bbi, weightID));
      if (!++bbi)
        break;
      TExample::iterator bii((*bi).begin()), bbii((*bbi).begin());
      for(diff = nattrs; diff && (*(bii++)==*(bbii++)); diff--);
    } while (!diff);

    bool hasUnknowns = hasSpecial(*bi);

    if (classDist->abs == 0.0 || hasUnknowns && !unknownsHandling)
      continue;

    TExample ex = *bi;
    ex.setClass(classVar->DK());
    ex.getClass().svalV = classDist;

    if (hasUnknowns) {
      if (unknownsHandling == UnknownsDistribute) {
        unknowns.addExample(ex);
        dynamic_cast<TDistribution &>(ex.getClass().svalV.getReference()) *= efmdata->getExampleWeight(ex);
        continue;
      }
      else
        classifier->containsUnknowns = true;
    }

    classifier->sortedExamples->addExample(ex);
  }

  if (unknowns.size()) {
    const int missWeight = getMetaID();

    efmdata = mlnew TEFMDataDescription(gen->domain, mlnew TDomainDistributions(gen), weightID, missWeight);
    wefmdata = efmdata;

    TExampleTable additionalExamples(gen->domain);

    EITERATE(ui, unknowns) {
      TExampleForMissing imputedExample(*ui, wefmdata);
      imputedExample.resetExample();
      do {
        additionalExamples.addExample(imputedExample);
        TExample &justAdded = additionalExamples.back();
        dynamic_cast<TDistribution &>(justAdded.getClass().svalV.getReference()) *= imputedExample.getMeta(missWeight).floatV;
        justAdded.removeMeta(missWeight);
      }
      while (imputedExample.nextExample());
    }

    PExampleGenerator wadde = PExampleGenerator(additionalExamples);
    TExampleTable sortedAdd(wadde, false);
    sortedAdd.sort();

    PExampleGenerator oldSortedExamples = classifier->sortedExamples;
    
    TExampleTable *sortedExamples = mlnew TExampleTable(gen->domain);
    classifier->sortedExamples = sortedExamples;

    for(TExampleIterator osi(oldSortedExamples->begin()), nsi(sortedAdd.begin()); osi && nsi; ) {
      int cmp = (*osi).compare(*nsi);
      if (cmp <= 0) {
        sortedExamples->addExample(*osi);
        ++osi;
      }
      else {
        TExample *lastAdded = sortedExamples->size() ? &sortedExamples->back() : NULL;
        if (lastAdded && !(*nsi).compare(*lastAdded))
          dynamic_cast<TDistribution &>(lastAdded->getClass().svalV.getReference()) += dynamic_cast<TDistribution &>((*nsi).getClass().svalV.getReference());
        else
          sortedExamples->addExample(*nsi);
        ++nsi;
      }
    }
  }

  if (learnerForUnknown)
    classifier->classifierForUnknown = learnerForUnknown->operator()(ogen, weightID);

  return wclassifier;
}



TClassifierByExampleTable::TClassifierByExampleTable(PDomain dom)
: TClassifierFD(dom),
  sortedExamples(mlnew TExampleTable(dom))
{}


TClassifierByExampleTable::TClassifierByExampleTable(PExampleGenerator gen, PClassifier unk)
: TClassifierFD(gen->domain),
  sortedExamples(mlnew TExampleTable(gen)),
  containsUnknowns(false),
  classifierForUnknown(unk)
{
  TFilter_hasSpecial hasSpecial;
  for(TExampleIterator ei(sortedExamples->begin()); ei && !containsUnknowns; containsUnknowns = hasSpecial(*ei), ++ei);
}



PDistribution TClassifierByExampleTable::classDistributionLow(const TExample &exam)
{
  TExample convertedEx(domain, exam);

  if (containsUnknowns || TFilter_hasSpecial()(convertedEx)) {
    bool weightUnknowns = dataDescription && (classVar->varType == TValue::INTVAR);
    TDistribution *distsum = TDistribution::create(classVar);
    PDistribution res = distsum;
    PEITERATE(ei, sortedExamples)
      if (convertedEx.compatible(*ei)) {
        TValue &classVal = (*ei).getClass();
        TDistribution *dist = classVal.svalV.AS(TDistribution);
        if (dist)
          if (weightUnknowns)
            ((TDiscDistribution *)(distsum))->adddist(*dist, dataDescription->getExampleWeight(*ei));
          else
            *distsum += *dist;
        else if (!classVal.isSpecial())
          distsum->addint(classVal.intV);
      }
    if (distsum->abs) {
      distsum->normalize();
      return res;
    }
    else
      return PDistribution();
  }

  int L = 0, H = sortedExamples->size();
  while(L<H) {
    const int M = (L+H)/2;
    int cmp = convertedEx.compare(sortedExamples->at(M));
    if (cmp > 0)
      L = M+1;
    else if (cmp > 0)
      H = M;
    else {
      TValue &classVal = sortedExamples->at(M).getClass();
      TDistribution *dist = classVal.svalV.AS(TDistribution);
      if (dist)
        return CLONE(TDistribution, dist);
      else {
        dist = TDistribution::create(classVar);
        dist->add(classVal);
        return dist;
      }  
    }
  }

  return PDistribution();
}


TValue TClassifierByExampleTable::operator()(const TExample &exam)
{ 
  PDistribution probs = classDistributionLow(exam);
  if (probs)
    return probs->highestProbValue(exam);
  else
    return classifierForUnknown ? classifierForUnknown->operator()(exam) : domain->classVar->DK();
}


PDistribution TClassifierByExampleTable::classDistribution(const  TExample &exam)
{ 
  PDistribution dval = classDistributionLow(exam);
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
{ 
  PDistribution dval = classDistributionLow(exam);
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
  if (   !strcmp(name, "sortedExamples")
      || !strcmp(name, "sorted_examples")) {
    domain = sortedExamples->domain; 
    classVar = sortedExamples->domain->classVar;
  }

  TClassifierFD::afterSet(name);
}
