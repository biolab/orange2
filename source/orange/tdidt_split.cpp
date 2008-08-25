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


#include "measures.hpp"
#include "random.hpp"
#include "classfromvar.hpp"
#include "discretize.hpp"
#include "table.hpp"

#include "boolcnt.hpp"
#include "tdidt.hpp"
#include "tdidt_stop.hpp"

#include "tdidt_split.ppp"


TTreeSplitConstructor::TTreeSplitConstructor(const float &aml)
: minSubset(aml>0 ? aml : 1e-20)
{}



TTreeSplitConstructor_Measure::TTreeSplitConstructor_Measure(PMeasureAttribute meas, const float &aworst, const float &aml)
: TTreeSplitConstructor(aml),
  measure(meas),
  worstAcceptable(aworst)
{}



TTreeSplitConstructor_Combined::TTreeSplitConstructor_Combined(PTreeSplitConstructor discreteSplit, PTreeSplitConstructor continuousSplit, const float &aminSubset)
: TTreeSplitConstructor(aminSubset),
  discreteSplitConstructor(discreteSplit),
  continuousSplitConstructor(continuousSplit)
{}



PClassifier TTreeSplitConstructor_Combined::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates,
                             PClassifier nodeClassifier
                            )
{ checkProperty(discreteSplitConstructor);
  checkProperty(continuousSplitConstructor);

  vector<bool> discrete, continuous;
 
  bool cse = candidates.size()==0;
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  TVarList::const_iterator vi(gen->domain->attributes->begin()), ve(gen->domain->attributes->end());

  for(; (cse || (ci!=ce)) && (vi!=ve); vi++) {
    if (cse || *(ci++))
      if ((*vi)->varType == TValue::INTVAR) {
        discrete.push_back(true);
        continuous.push_back(false);
        continue;
      }
      else if ((*vi)->varType == TValue::FLOATVAR) {
        discrete.push_back(false);
        continuous.push_back(true);
        continue;
      }
    discrete.push_back(false);
    continuous.push_back(false);
  }

  float discQuality;
  PStringList discDescriptions;
  PDiscDistribution discSizes;
  int discSpent;
  PClassifier discSplit = discreteSplitConstructor->call(discDescriptions, discSizes, discQuality, discSpent,
                                                               gen, weightID, dcont, apriorClass, discrete, nodeClassifier);

  float contQuality;
  PStringList contDescriptions;
  PDiscDistribution contSizes;
  int contSpent;
  PClassifier contSplit = continuousSplitConstructor->call(contDescriptions, contSizes, contQuality, contSpent,
                                                                 gen, weightID, dcont, apriorClass, continuous, nodeClassifier);

  int N = gen ? gen->numberOfExamples() : -1;
  if (N<0)
    N = dcont->classes->cases;

  if (   discSplit
      && (   !contSplit
          || (discQuality>contQuality)
          || (discQuality==contQuality) && (N%2>0))) {
    quality = discQuality;
    descriptions = discDescriptions;
    subsetSizes = discSizes;
    spentAttribute = discSpent;
    return discSplit;
  }
  else if (contSplit) {
    quality = contQuality;
    descriptions = contDescriptions;
    subsetSizes = contSizes;
    spentAttribute = contSpent;
    return contSplit;
  }
  else 
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);
}



TTreeSplitConstructor_Attribute::TTreeSplitConstructor_Attribute(PMeasureAttribute meas, const float &worst, const float &ms)
: TTreeSplitConstructor_Measure(meas, worst, ms)
{}


// rejects the split if there are less than two non-empty branches
// or there is a non-empty branch with less then minSubset examples
bool checkDistribution(const TDiscDistribution &dist, const float &minSubset)
{
  int nonzero = 0;
  for(TDiscDistribution::const_iterator dvi(dist.begin()), dve(dist.end()); dvi!=dve; dvi++)
    if (*dvi > 0) {
      if  (*dvi < minSubset)
        return false;
      nonzero++;
    }

  return nonzero >= 2;
}



inline bool noCandidates(const vector<bool> &candidates)
{
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  while(ci!=ce && !*ci)
    ci++;
  return ci==ce;
}

PClassifier TTreeSplitConstructor_Attribute::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates,
                             PClassifier nodeClassifier
                            )
{ checkProperty(measure);

  measure->checkClassTypeExc(gen->domain->classVar->varType);

  bool cse = candidates.size()==0;
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  if (!cse) {
    if (noCandidates(candidates))
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    ci = candidates.begin();
  }

  int N = gen ? gen->numberOfExamples() : -1;
  if (N<0)
    N = dcont->classes->cases;
  TSimpleRandomGenerator rgen(N);

  int thisAttr = 0, bestAttr = -1, wins = 0;
  quality = 0.0;

  if (measure->needs == TMeasureAttribute::Contingency_Class) {
    vector<bool> myCandidates;
    if (cse) {
      myCandidates.reserve(gen->domain->attributes->size());
      PITERATE(TVarList, vi, gen->domain->attributes)
        myCandidates.push_back((*vi)->varType == TValue::INTVAR);
    }
    else {
      myCandidates.reserve(candidates.size());
      TVarList::const_iterator vi(gen->domain->attributes->begin());
      for(; ci != ce; ci++, vi++)
        myCandidates.push_back(*ci && ((*vi)->varType == TValue::INTVAR));
    }

    if (!dcont || dcont->classIsOuter)
      dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID, myCandidates));

    ci = myCandidates.begin();
    ce = myCandidates.end();
    TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
    for(; (ci != ce) && (dci!=dce); dci++, ci++, thisAttr++)
      if (*ci && checkDistribution((const TDiscDistribution &)((*dci)->outerDistribution.getReference()), minSubset)) {
        float thisMeas = measure->call(thisAttr, dcont, apriorClass);

        if (   ((!wins || (thisMeas>quality)) && ((wins=1)==1))
            || ((thisMeas==quality) && rgen.randbool(++wins))) {
          quality = thisMeas;
          subsetSizes = (*dci)->outerDistribution;
          bestAttr = thisAttr;
        }
      }
  }

  else if (measure->needs == TMeasureAttribute::DomainContingency) {
    if (!dcont || dcont->classIsOuter)
      dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID));

    TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
    for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++)
      if (    (cse || *(ci++))
           && ((*dci)->outerVariable->varType==TValue::INTVAR)
           && checkDistribution((const TDiscDistribution &)((*dci)->outerDistribution.getReference()), minSubset)) {
        float thisMeas = measure->call(thisAttr, dcont, apriorClass);

        if (   ((!wins || (thisMeas>quality)) && ((wins=1)==1))
            || ((thisMeas==quality) && rgen.randbool(++wins))) {
          quality = thisMeas;
          subsetSizes = (*dci)->outerDistribution;
          bestAttr = thisAttr;
        }
      }
  }

  else {
    TDomainDistributions ddist(gen, weightID);

    TDomainDistributions::iterator ddi(ddist.begin()), dde(ddist.end()-1);
    for(; (cse || (ci!=ce)) && (ddi!=dde); ddi++, thisAttr++)
      if (cse || *(ci++)) {
        TDiscDistribution *discdist = (*ddi).AS(TDiscDistribution);
        if (discdist && checkDistribution(*discdist, minSubset)) {
          float thisMeas = measure->call(thisAttr, gen, apriorClass, weightID);

          if (   ((!wins || (thisMeas>quality)) && ((wins=1)==1))
              || ((thisMeas==quality) && rgen.randbool(++wins))) {
            quality = thisMeas;
            subsetSizes = PDiscDistribution(*ddi); // not discdist - this would be double wrapping!
            bestAttr = thisAttr;
          }
        }
      }
    
  }

  if (!wins)
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  if (quality<worstAcceptable)
    return returnNothing(descriptions, subsetSizes, spentAttribute);

  PVariable attribute = gen->domain->attributes->at(bestAttr);
  TEnumVariable *evar = attribute.AS(TEnumVariable);
  if (evar)
    descriptions = mlnew TStringList(evar->values.getReference());
  else
    descriptions = mlnew TStringList(subsetSizes->size(), "");

  spentAttribute = bestAttr;

  return mlnew TClassifierFromVarFD(attribute, gen->domain, bestAttr, subsetSizes);
}




TTreeSplitConstructor_ExhaustiveBinary::TTreeSplitConstructor_ExhaustiveBinary(PMeasureAttribute meas, const float &aworst, const float &aml)
: TTreeSplitConstructor_Measure(meas, aworst, aml)
{}



PClassifier TTreeSplitConstructor_ExhaustiveBinary::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates,
                             PClassifier
                            )
{ 
  checkProperty(measure);
  measure->checkClassTypeExc(gen->domain->classVar->varType);

  PIntList bestMapping;
  int wins, bestAttr;
  PVariable bvar;

  if (measure->needs==TMeasureAttribute::Generator) {
    bool cse = candidates.size()==0;
    bool haveCandidates = false;
    vector<bool> myCandidates;
    myCandidates.reserve(gen->domain->attributes->size());
    vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
    TVarList::const_iterator vi, ve(gen->domain->attributes->end());
    for(vi = gen->domain->attributes->begin(); vi != ve; vi++) {
      bool co = (*vi)->varType == TValue::INTVAR && (!cse || (ci!=ce) && *ci);
      myCandidates.push_back(co);
      haveCandidates = haveCandidates || co;
    }
    if (!haveCandidates)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    PDistribution thisSubsets;
    float thisQuality;
    wins = 0;
    int thisAttr = 0;

    int N = gen->numberOfExamples();
    TSimpleRandomGenerator rgen(N);

    ci = myCandidates.begin();
    for(vi = gen->domain->attributes->begin(); vi != ve; ci++, vi++, thisAttr++) {
      if (*ci) {
        thisSubsets = NULL;
        PIntList thisMapping =
           /*throughCont ? measure->bestBinarization(thisSubsets, thisQuality, *dci, dcont->classes, apriorClass, minSubset)
                       : */measure->bestBinarization(thisSubsets, thisQuality, *vi, gen, apriorClass, weightID, minSubset);
          if (thisMapping
                && (   (!wins || (thisQuality>quality)) && ((wins=1)==1)
                    || (thisQuality==quality) && rgen.randbool(++wins))) {
            bestAttr = thisAttr;
            quality = thisQuality;
            subsetSizes = thisSubsets;
            bestMapping = thisMapping;
          }
      }
      /*if (thoughCont)
        dci++; */
    }
  
    if (!wins)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    if (quality<worstAcceptable)
      return returnNothing(descriptions, subsetSizes, spentAttribute);

    if (subsetSizes && subsetSizes->variable)
      bvar = subsetSizes->variable;
    else {
      TEnumVariable *evar = mlnew TEnumVariable("");
      evar->addValue("0");
      evar->addValue("1");
      bvar = evar;
    }
  }
  
  else {
    bool cse = candidates.size()==0;
    if (!cse && noCandidates(candidates))
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    if (!dcont || dcont->classIsOuter) {
      dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID));
      raiseWarningWho("TreeSplitConstructor_ExhaustiveBinary", "this class is not optimized for 'candidates' list and can be very slow");
    }

    int N = gen ? gen->numberOfExamples() : -1;
    if (N<0)
      N = dcont->classes->cases;
    TSimpleRandomGenerator rgen(N);

    PDistribution classDistribution = dcont->classes;

    vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());

    TDiscDistribution *dis0, *dis1;
    TContDistribution *con0, *con1;

    int thisAttr = 0;
    bestAttr = -1;
    wins = 0;
    quality = 0.0;
    float leftExamples, rightExamples;

    TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
    for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++) {

      // We consider the attribute only if it is a candidate, discrete and has at least two values
      if ((cse || *(ci++)) && ((*dci)->outerVariable->varType==TValue::INTVAR) && ((*dci)->discrete->size()>=2)) {

        const TDistributionVector &distr = *(*dci)->discrete;

        if (distr.size()>10)
          raiseError("'%s' has more than 10 values, cannot exhaustively binarize", gen->domain->attributes->at(thisAttr)->name.c_str());

        // If the attribute is binary, we check subsetSizes and assess the quality if they are OK
        if (distr.size()==2) {
          if ((distr.front()->abs<minSubset) || (distr.back()->abs<minSubset))
            continue; // next attribute
          else {
            float thisMeas = measure->call(thisAttr, dcont, apriorClass);
            if (   ((!wins || (thisMeas>quality)) && ((wins=1)==1))
                || ((thisMeas==quality) && rgen.randbool(++wins))) {
              bestAttr = thisAttr;
              quality = thisMeas;
              leftExamples = distr.front()->abs;
              rightExamples = distr.back()->abs;
              bestMapping = mlnew TIntList(2, 0);
              bestMapping->at(1) = 1;
            }
            continue;
          }
        }

        vector<int> valueIndices;
        int ind = 0;
        for(TDistributionVector::const_iterator dvi(distr.begin()), dve(distr.end()); (dvi!=dve); dvi++, ind++)
          if ((*dvi)->abs>0)
            valueIndices.push_back(ind);

        if (valueIndices.size()<2)
          continue;

        PContingency cont = prepareBinaryCheat(classDistribution, *dci, bvar, dis0, dis1, con0, con1);

        // A real job: go through all splits
        int binWins = 0;
        float binQuality = -1.0;
        float binLeftExamples = -1.0, binRightExamples = -1.0;
        // Selection: each element correspons to a value of the original attribute and is 1, if the value goes right
        // The first value always goes left (and has no corresponding bit in selection.
        TBoolCount selection(valueIndices.size()-1), bestSelection(0);

        // First for discrete classes
        if (dis0) {
          do {
            *dis0 = CAST_TO_DISCDISTRIBUTION(distr[valueIndices[0]]);
            *dis1 *= 0;
            vector<int>::const_iterator ii(valueIndices.begin());
            for(TBoolCount::const_iterator bi(selection.begin()), be(selection.end()); bi!=be; bi++, ii++)
               *(*bi ? dis1 : dis0) += distr[*ii];

            if ((dis0->abs<minSubset) || (dis1->abs<minSubset))
              continue; // cannot split like that, to few examples in one of the branches

            float thisMeas = measure->operator()(cont, classDistribution, apriorClass);
            if (   ((!binWins) || (thisMeas>binQuality)) && ((binWins=1) ==1)
                || (thisMeas==binQuality) && rgen.randbool(++binWins)) {
              bestSelection = selection; 
              binQuality = thisMeas;
              binLeftExamples = dis0->abs;
              binRightExamples = dis1->abs;
            }
          } while (selection.next());
        }

        // And then exactly the same for continuous classes
        else {
          do {
            *con0 = CAST_TO_CONTDISTRIBUTION(distr[0]);
            *con1 = TContDistribution();
            vector<int>::const_iterator ii(valueIndices.begin());
            for(TBoolCount::const_iterator bi(selection.begin()), be(selection.end()); bi!=be; bi++, ii++)
               *(*bi ? con1 : con0) += distr[*ii];

            if ((con0->abs<minSubset) || (con1->abs<minSubset))
              continue; // cannot split like that, to few examples in one of the branches

            float thisMeas = measure->operator()(cont, classDistribution, apriorClass);
            if (   ((!binWins) || (thisMeas>binQuality)) && ((binWins=1) ==1)
                || (thisMeas==binQuality) && rgen.randbool(++binWins)) {
              bestSelection = selection; 
              binQuality = thisMeas;
              binLeftExamples = con0->abs;
              binRightExamples = con1->abs;
            }
          } while (selection.next());
        }

        if (       binWins
            && (   (!wins || (binQuality>quality)) && ((wins=1)==1)
                || (binQuality==quality) && rgen.randbool(++wins))) {
          bestAttr = thisAttr;
          quality = binQuality;
          leftExamples = binLeftExamples;
          rightExamples = binRightExamples;
          bestMapping = mlnew TIntList(distr.size(), -1);
          vector<int>::const_iterator ii = valueIndices.begin();
          bestMapping->at(*(ii++)) = 0;
          ITERATE(TBoolCount, bi, selection)
            bestMapping->at(*(ii++)) = *bi ? 1 : 0;
        }
      }
    }
 

    if (!wins)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    subsetSizes = mlnew TDiscDistribution();
    subsetSizes->addint(0, leftExamples);
    subsetSizes->addint(1, rightExamples);
  }

  PVariable attribute = gen->domain->attributes->at(bestAttr);

  if (attribute->noOfValues() == 2) {
    spentAttribute = bestAttr;
    descriptions = mlnew TStringList(attribute.AS(TEnumVariable)->values.getReference());
    return mlnew TClassifierFromVarFD(attribute, gen->domain, bestAttr, subsetSizes);
  }

  string s0, s1;
  int ns0 = 0, ns1 = 0;
  TValue ev;
  attribute->firstValue(ev);
  PITERATE(TIntList, mi, bestMapping) {
    string str;
    attribute->val2str(ev, str);
    if (*mi==1) {
      s1 += string(ns1 ? ", " : "") + str;
      ns1++;
    }
    else if (*mi==0) {
      s0 += string(ns0 ? ", " : "") + str;
      ns0++;
    }

    attribute->nextValue(ev);
  }

  descriptions = mlnew TStringList();
  descriptions->push_back(ns0>1 ? "in ["+s0+"]" : s0);
  descriptions->push_back(ns1>1 ? "in ["+s1+"]" : s1);

  bvar->name = gen->domain->attributes->at(bestAttr)->name;
  spentAttribute = (ns0==1) && (ns1==1) ? bestAttr : -1;
  return mlnew TClassifierFromVarFD(bvar, gen->domain, bestAttr, subsetSizes, mlnew TMapIntValue(bestMapping));
}









PClassifier TTreeSplitConstructor_OneAgainstOthers::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates,
                             PClassifier
                            )
{ 
  checkProperty(measure);
  measure->checkClassTypeExc(gen->domain->classVar->varType);

  int bestValue, wins, bestAttr;
  PVariable bvar;

  if (measure->needs==TMeasureAttribute::Generator) {
    bool cse = candidates.size()==0;
    bool haveCandidates = false;
    vector<bool> myCandidates;
    myCandidates.reserve(gen->domain->attributes->size());
    vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
    TVarList::const_iterator vi, ve(gen->domain->attributes->end());
    for(vi = gen->domain->attributes->begin(); vi != ve; vi++) {
      bool co = (*vi)->varType == TValue::INTVAR && (!cse || (ci!=ce) && *ci);
      myCandidates.push_back(co);
      haveCandidates = haveCandidates || co;
    }
    if (!haveCandidates)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    PDistribution thisSubsets;
    float thisQuality;
    wins = 0;
    int thisAttr = 0;

    int N = gen->numberOfExamples();
    TSimpleRandomGenerator rgen(N);

    ci = myCandidates.begin();
    for(vi = gen->domain->attributes->begin(); vi != ve; ci++, vi++, thisAttr++) {
      if (*ci) {
        thisSubsets = NULL;
        int thisValue = measure->bestValue(thisSubsets, thisQuality, *vi, gen, apriorClass, weightID, minSubset);
        if ((thisValue >=0)
                && (   (!wins || (thisQuality>quality)) && ((wins=1)==1)
                    || (thisQuality==quality) && rgen.randbool(++wins))) {
            bestAttr = thisAttr;
            quality = thisQuality;
            subsetSizes = thisSubsets;
            bestValue = thisValue;
          }
      }
    }
  
    if (!wins)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    if (quality<worstAcceptable)
      return returnNothing(descriptions, subsetSizes, spentAttribute);

    if (subsetSizes && subsetSizes->variable)
      bvar = subsetSizes->variable;
    else {
      TEnumVariable *evar = mlnew TEnumVariable("");
      const string &value = gen->domain->attributes->at(bestAttr).AS(TEnumVariable)->values->at(bestValue);
      evar->addValue(string("not ")+value);
      evar->addValue(value);
      bvar = evar;
    }
  }
  
  else {
    bool cse = candidates.size()==0;
    if (!cse && noCandidates(candidates))
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    if (!dcont || dcont->classIsOuter) {
      dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID));
    }

    int N = gen ? gen->numberOfExamples() : -1;
    if (N<0)
      N = dcont->classes->cases;
    TSimpleRandomGenerator rgen(N);

    PDistribution classDistribution = dcont->classes;

    vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());

    TDiscDistribution *dis0, *dis1;
    TContDistribution *con0, *con1;

    int thisAttr = 0;
    bestAttr = -1;
    wins = 0;
    quality = 0.0;
    float leftExamples, rightExamples;

    TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
    for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++) {

      // We consider the attribute only if it is a candidate, discrete and has at least two values
      if ((cse || *(ci++)) && ((*dci)->outerVariable->varType==TValue::INTVAR) && ((*dci)->discrete->size()>=2)) {

        const TDistributionVector &distr = *(*dci)->discrete;

        // If the attribute is binary, we check subsetSizes and assess the quality if they are OK
        if (distr.size() == 2) {
          if ((distr.front()->abs < minSubset) || (distr.back()->abs < minSubset))
            continue; // next attribute
          else {
            float thisMeas = measure->call(thisAttr, dcont, apriorClass);
            if (   ((!wins || (thisMeas>quality)) && ((wins=1)==1))
                || ((thisMeas==quality) && rgen.randbool(++wins))) {
              bestAttr = thisAttr;
              quality = thisMeas;
              leftExamples = distr.front()->abs;
              rightExamples = distr.back()->abs;
              bestValue = 1;
            }
            continue;
          }
        }

        int binWins = 0, binBestValue = -1;
        float binQuality = -1.0;
        float binLeftExamples = -1.0, binRightExamples = -1.0;

        PContingency cont = prepareBinaryCheat(classDistribution, *dci, bvar, dis0, dis1, con0, con1);
        int thisValue = 0;
        const float maxSubset = (*dci)->innerDistribution->abs - minSubset;
        for(TDistributionVector::const_iterator dvi(distr.begin()), dve(distr.end()); (dvi!=dve); dvi++, thisValue++) {
          if (((*dvi)->abs < minSubset) || ((*dvi)->abs > maxSubset))
            continue;

          float thisMeas;
          
          // First for discrete classes
          if (dis0) {
            *dis0 = CAST_TO_DISCDISTRIBUTION(*dvi);
            *dis1 = CAST_TO_DISCDISTRIBUTION((*dci)->innerDistribution);
            *dis1 -= *dis0;
            thisMeas = measure->operator()(cont, classDistribution, apriorClass);
          }
          else {
            *con0 = CAST_TO_CONTDISTRIBUTION(*dvi);
            *con1 = CAST_TO_CONTDISTRIBUTION((*dci)->innerDistribution);
            *con0 -= *con1;
            thisMeas = measure->operator()(cont, classDistribution, apriorClass);
          }
          
          if (   ((!binWins) || (thisMeas>binQuality)) && ((binWins=1) ==1)
              || (thisMeas==binQuality) && rgen.randbool(++binWins)) {
            binBestValue = thisValue; 
            binQuality = thisMeas;
            binLeftExamples = dis0->abs;
            binRightExamples = dis1->abs;
          }
        }

        if (       binWins
            && (   (!wins || (binQuality>quality)) && ((wins=1)==1)
                || (binQuality==quality) && rgen.randbool(++wins))) {
          bestAttr = thisAttr;
          quality = binQuality;
          leftExamples = binLeftExamples;
          rightExamples = binRightExamples;
          bestValue = binBestValue;
        }
      }
    }
 

    if (!wins)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

    subsetSizes = mlnew TDiscDistribution();
    subsetSizes->addint(0, leftExamples);
    subsetSizes->addint(1, rightExamples);
  }

  PVariable attribute = gen->domain->attributes->at(bestAttr);

  if (attribute->noOfValues() == 2) {
    spentAttribute = bestAttr;
    descriptions = mlnew TStringList(attribute.AS(TEnumVariable)->values.getReference());
    return mlnew TClassifierFromVarFD(attribute, gen->domain, bestAttr, subsetSizes);
  }

  const string &bestValueS = attribute.AS(TEnumVariable)->values->at(bestValue);
  descriptions = mlnew TStringList();
  descriptions->push_back(string("not ") + bestValueS);
  descriptions->push_back(bestValueS);
  
  bvar->name = gen->domain->attributes->at(bestAttr)->name;

  TIntList *bestMapping = mlnew TIntList(attribute.AS(TEnumVariable)->values->size(), 0);
  PIntList wb = bestMapping;
  bestMapping->at(bestValue) = 1;
  spentAttribute = -1;
  return mlnew TClassifierFromVarFD(bvar, gen->domain, bestAttr, subsetSizes, mlnew TMapIntValue(bestMapping));
}










TTreeSplitConstructor_Threshold::TTreeSplitConstructor_Threshold(PMeasureAttribute meas, const float &worst, const float &aml)
: TTreeSplitConstructor_Measure(meas, worst, aml)
{}


PClassifier TTreeSplitConstructor_Threshold::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates,
                             PClassifier
                            )
{ 
  checkProperty(measure);
  measure->checkClassTypeExc(gen->domain->classVar->varType);

  bool cse = candidates.size()==0;
  bool haveCandidates = false;
  vector<bool> myCandidates;
  myCandidates.reserve(gen->domain->attributes->size());
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  TVarList::const_iterator vi, ve(gen->domain->attributes->end());
  for(vi = gen->domain->attributes->begin(); vi != ve; vi++) {
    bool co = (*vi)->varType == TValue::FLOATVAR && (cse || (ci!=ce) && *ci);
    if (ci != ce)
      ci++;
    myCandidates.push_back(co);
    haveCandidates = haveCandidates || co;
  }
  if (!haveCandidates)
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  int N = gen ? gen->numberOfExamples() : -1;
  if (N < 0)
    N = dcont->classes->cases;

  TSimpleRandomGenerator rgen(N);


  PDistribution thisSubsets;
  float thisQuality, bestThreshold;
  ci = myCandidates.begin();
  int wins = 0, thisAttr = 0, bestAttr;

  TDomainContingency::iterator dci, dce;
  bool throughCont = (dcont && !dcont->classIsOuter && (measure->needs <= measure->DomainContingency));
  if (throughCont) {
    dci = dcont->begin();
    dce = dcont->end();
  }

  for(vi = gen->domain->attributes->begin(); vi != ve; ci++, vi++, thisAttr++) {
    if (*ci) {
      thisSubsets = NULL;
      const float thisThreshold =
         throughCont ? measure->bestThreshold(thisSubsets, thisQuality, *dci, dcont->classes, apriorClass, minSubset)
                     : measure->bestThreshold(thisSubsets, thisQuality, *vi, gen, apriorClass, weightID, minSubset);
        if ((thisThreshold != ILLEGAL_FLOAT)
              && (   (!wins || (thisQuality>quality)) && ((wins=1)==1)
                  || (thisQuality==quality) && rgen.randbool(++wins))) {
          bestAttr = thisAttr;
          quality = thisQuality;
          subsetSizes = thisSubsets;
          bestThreshold = thisThreshold;
        }
    }
    if (throughCont)
      dci++;
  }
  
  if (!wins)
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  if (quality<worstAcceptable)
    return returnNothing(descriptions, subsetSizes, spentAttribute);

  PVariable bvar;
  if (subsetSizes && subsetSizes->variable)
    bvar = subsetSizes->variable;
  else {
    TEnumVariable *evar = mlnew TEnumVariable("");
    evar->addValue("0");
    evar->addValue("1");
    bvar = evar;
  }

  descriptions = mlnew TStringList();
  char str[128];
  sprintf(str, "<=%3.3f", bestThreshold);
  descriptions->push_back(str);
  sprintf(str, ">%3.3f", bestThreshold);
  descriptions->push_back(str);

  bvar->name = gen->domain->attributes->at(bestAttr)->name;
  spentAttribute = -1;
  return mlnew TClassifierFromVarFD(bvar, gen->domain, bestAttr, subsetSizes, mlnew TThresholdDiscretizer(bestThreshold));
}



PExampleGeneratorList TTreeExampleSplitter::prepareGeneratorList(int size, PExampleGenerator gen, vector<TExampleTable *> &unwrapped)
{
  PExampleTable lock = gen.AS(TExampleTable);
  if (lock) {
    if (lock->lock)
      lock = lock->lock;
  }
  else {
    lock = mlnew TExampleTable(gen);
  }
    
  PExampleGeneratorList examplePtrs = mlnew TExampleGeneratorList();
  while(size--) {
    TExampleTable *ntable = mlnew TExampleTable(lock, 1);
    examplePtrs->push_back(PExampleGenerator(ntable));
    unwrapped.push_back(ntable);
  }

  return examplePtrs;
}


bool TTreeExampleSplitter::getBranchIndices(PTreeNode node, PExampleGenerator gen, vector<int> &indices)
{
  TClassifier &branchSelector = node->branchSelector.getReference();
  const int maxIndex = node->branchDescriptions->size();
  
  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (index.isSpecial() || (index.intV<0) || (index.intV>=maxIndex))
      return false;
    indices.push_back(index.intV);
  }

  return true;
}

PExampleGeneratorList TTreeExampleSplitter_IgnoreUnknowns::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = node->branchSelector.getReference();
  const int maxIndex = node->branchDescriptions->size();

  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);
  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToCommon::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ 
  if (!node->branchSizes)
    raiseError("TreeExampleSplitter_UnknownsToCommon: splitConstructor didn't set the branchSize; use different constructor or splitter");

  TClassifier &branchSelector = node->branchSelector.getReference();
  const int maxIndex = node->branchDescriptions->size();
  const int mostCommon = node->branchSizes->highestProbIntIndex();

  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    uexamplePtrs[!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex) ? index.intV : mostCommon]->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToAll::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = node->branchSelector.getReference();
  const int maxIndex = node->branchDescriptions->size();

  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
    else
      ITERATE(vector<TExampleTable *>, pei, uexamplePtrs)
        (*pei)->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToRandom::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = node->branchSelector.getReference();
  const int maxIndex = node->branchDescriptions->size();

  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
    else {
      TDiscDistribution *distr = NULL;
      if (index.svalV)
        distr = index.svalV.AS(TDiscDistribution);
      if (!distr)
        distr = node->branchSizes.AS(TDiscDistribution);
      if (distr)
        uexamplePtrs[distr->randomInt()]->addExample(*ei);
    }
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToBranch::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = node->branchSelector.getReference();
  int maxIndex = node->branchDescriptions->size();
  node->branchDescriptions->push_back("unknown");

  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex+1, gen, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
    else
      uexamplePtrs.back()->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsAsBranchSizes::operator()(PTreeNode node, PExampleGenerator gen, const int &weightID, vector<int> &newWeights)
{ 
  int maxIndex = node->branchDescriptions->size();
  TClassifier &branchSelector = node->branchSelector.getReference();
 
  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);

  vector<int> indices;
  
  if (getBranchIndices(node, gen, indices)) {
    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      ++ei;
    }
  }

  else {
    if (!node->branchSizes)
      raiseError("TreeExampleSplitter_UnknownsAsBranchSizes: splitConstructor didn't set the branchSize; use different constructor or splitter");

    const TDiscDistribution &branchSizes = node->branchSizes.getReference();
    for(int i = maxIndex; i--; )
      newWeights.push_back(getMetaID());

    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      (*ei).setMeta(newWeights[*ii], TValue(WEIGHT(*ei)));
      ++ei;
    }

    for (; ei; ++ei) {
      TValue index = branchSelector(*ei);

      if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex)) {
        uexamplePtrs[index.intV]->addExample(*ei);
        (*ei).setMeta(newWeights[index.intV], TValue(WEIGHT(*ei)));
      }
    
      else {
        if (index.isDC()) {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            uexamplePtrs[branchNo]->addExample(*ei);
            (*ei).setMeta(newWeights[branchNo], TValue(WEIGHT(*ei)));
          }
        }
        else {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            float weight = branchSizes.p(branchNo) * WEIGHT(*ei);
            if (weight) {
              uexamplePtrs[branchNo]->addExample(*ei);
              (*ei).setMeta(newWeights[branchNo], TValue(weight));
            }
          }
        }
      }
    }
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsAsSelector::operator()(PTreeNode node, PExampleGenerator gen, const int &weightID, vector<int> &newWeights)
{ TClassifier &branchSelector = node->branchSelector.getReference();
  int maxIndex = node->branchDescriptions->size();

 
  vector<TExampleTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen, uexamplePtrs);

  vector<int> indices;
  
  if (getBranchIndices(node, gen, indices)) {
    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      ++ei;
    }
  }

  else {
    for(int i = maxIndex; i--; )
      newWeights.push_back(getMetaID());

    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      (*ei).setMeta(newWeights[*ii], TValue(WEIGHT(*ei)));
      ++ei;
    }

    for (; ei; ++ei) {
      TValue index = branchSelector(*ei);

      if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex)) {
        uexamplePtrs[index.intV]->addExample(*ei);
        (*ei).setMeta(newWeights[index.intV], TValue(WEIGHT(*ei)));
      }
    
      else {
        if (index.isDC()) {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            uexamplePtrs[branchNo]->addExample(*ei);
            (*ei).setMeta(newWeights[branchNo], TValue(WEIGHT(*ei)));
          }
        }
        else {
          TDiscDistribution *distr = index.svalV ? index.svalV.AS(TDiscDistribution) : NULL;
          if (distr)
            for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
              float weight = distr->p(branchNo) * WEIGHT(*ei);
              if (weight) {
                uexamplePtrs[branchNo]->addExample(*ei);
                (*ei).setMeta(newWeights[branchNo], TValue(weight));
            }
          }
        }
      }
    }
  }

  return examplePtrs;
}
