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


TMapIntValue::TMapIntValue(PIntList al)
: mapping(al)
{}


TMapIntValue::TMapIntValue(const TIntList &al)
: mapping(mlnew TIntList(al))
{}


void TMapIntValue::transform(TValue &val)
{ checkProperty(mapping);

  if (val.isSpecial())
    return;
  if (val.varType!=TValue::INTVAR)
    raiseErrorWho("transform", "invalid value type (discrete expected)");
  if (val.intV>=int(mapping->size()))
    raiseErrorWho("transform", "value out of range");

  int res = mapping->at(val.intV);
  if (res<0)
    val.setDK();
  else
    val.intV = res;
}



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
                             const vector<bool> &candidates
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
                                                               gen, weightID, dcont, apriorClass, discrete);

  float contQuality;
  PStringList contDescriptions;
  PDiscDistribution contSizes;
  int contSpent;
  PClassifier contSplit = continuousSplitConstructor->call(contDescriptions, contSizes, contQuality, contSpent,
                                                                 gen, weightID, dcont, apriorClass, continuous);

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


PClassifier TTreeSplitConstructor_Attribute::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates
                            )
{ checkProperty(measure);

  measure->checkClassTypeExc(gen->domain->classVar->varType);

  bool cse = candidates.size()==0;
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  if (!cse) {
    while(ci!=ce && !*ci)
      ci++;
    if (ci==ce)
      return returnNothing(descriptions, subsetSizes, quality, spentAttribute);
    ci = candidates.begin();
  }

  int N = gen ? gen->numberOfExamples() : -1;
  if (N<0)
    N = dcont->classes->cases;
  TSimpleRandomGenerator rgen(N);

  int thisAttr = 0, bestAttr = -1, wins = 0;
  quality = 0.0;

  if (measure->needs < TMeasureAttribute::Generator) {
    if (!dcont || dcont->classIsOuter)
      dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID));

    TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
    for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++)
      if ((cse || *(ci++)) && ((*dci)->outerVariable->varType==TValue::INTVAR)) {
        // if there is a non-null branch with less than minSubset examples, skip the attribute
        //   (the attribute has a value which is present but not with enough examples)
        // Also, if the attribute's value is same for all examples, attribute is useless and skipped
        int nonzero = 0;
        TDistributionVector::const_iterator dvi((*dci)->discrete->begin()), dve((*dci)->discrete->end());
        for(; (dvi!=dve) && (((*dvi)->abs==0) || ((*dvi)->abs>=minSubset)); dvi++)
          if ((*dvi)->abs>0)
            nonzero++;

        if ((dvi!=dve) || (nonzero<2))
          continue;

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
        if (discdist) {
          // if there is a non-null branch with less than minSubset examples, skip the attribute
          //   (the attribute has a value which is present but not with enough examples)
          // Also, if the attribute's value is same for all examples, attribute is unuseful and skipped
          int nonzero = 0;
          TDiscDistribution::const_iterator dvi(discdist->begin()), dve(discdist->end());
          for(; (dvi!=dve) && ((*dvi==0) || (*dvi>=minSubset)); dvi++)
            if (*dvi>0)
              nonzero++;

          if ((dvi!=dve) || (nonzero<2))
            continue;

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


inline bool noCandidates(const vector<bool> &candidates)
{
  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());
  while(ci!=ce && !*ci)
    ci++;
  return ci==ce;
}


/* Prepares the common stuff for binarization in classification trees:
   - a binary attribute 
   - a contingency matrix for this attribute
   - a DomainContingency that contains this matrix at position newpos (the last)
   - dis0 and dis1 (or con0 and con1, if the class is continuous) that point to distributions
     for the left and the right branch
*/
void prepareBinaryCheat(PExampleGenerator gen, const int &weightID, PVariable &bvar, PDomainContingency &dcont, int &newpos,
                        TDiscDistribution *&dis0, TDiscDistribution *&dis1, TContDistribution *&con0, TContDistribution *&con1)
{
  if (!dcont || dcont->classIsOuter)
    dcont = PDomainContingency(mlnew TDomainContingency(gen, weightID));

  TEnumVariable *ebvar = mlnew TEnumVariable("");
  ebvar->addValue("0");
  ebvar->addValue("1");
  bvar = ebvar;

  /* An ugly cheat that is prone to cause problems when Contingency class is changed.
     It is fast, though :) */
  TContingencyClass *cont = mlnew TContingencyAttrClass(bvar, dcont->classes->variable);
  dcont->push_back(cont);
  cont->innerDistribution = dcont->classes;
  cont->operator[](1);
  newpos = dcont->size()-1;

  if (dcont->classes->variable->varType==TValue::INTVAR) {
    dis0 = cont->discrete->front().AS(TDiscDistribution);
    dis1 = cont->discrete->back().AS(TDiscDistribution);
    con0 = con1 = NULL;
  }
  else {
    con0 = cont->discrete->front().AS(TContDistribution);
    con1 = cont->discrete->back().AS(TContDistribution);
    dis0 = dis1 = NULL;
  }
}


PClassifier TTreeSplitConstructor_ExhaustiveBinary::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates
                            )
{ 
  checkProperty(measure);

  // This we cannot offer: we would need to go through all examples each time and the attribute measure (say Relief) as well. Too slow.
  if (measure->needs==TMeasureAttribute::Generator)
    raiseError("cannot use a measure that requires example set");
  
  measure->checkClassTypeExc(gen->domain->classVar->varType);

  bool cse = candidates.size()==0;
  if (!cse && noCandidates(candidates))
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  int N = gen ? gen->numberOfExamples() : -1;
  if (N<0)
    N = dcont->classes->cases;
  TSimpleRandomGenerator rgen(N);

  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());

  PVariable bvar;
  int newpos;
  TDiscDistribution *dis0, *dis1;
  TContDistribution *con0, *con1;
  prepareBinaryCheat(gen, weightID, bvar, dcont, newpos, dis0, dis1, con0, con1);

  int thisAttr = 0, bestAttr = -1, wins = 0;
  quality = 0.0;
  float leftExamples, rightExamples;
  PIntList bestMapping;

  TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
  for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++)
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

          float thisMeas = measure->operator()(newpos, dcont, apriorClass);
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

          float thisMeas = measure->operator()(newpos, dcont, apriorClass);
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
 

  dcont->erase(dcont->begin()+newpos); // removes the added contingency from the domain contingency
  if (!wins)
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  subsetSizes = mlnew TDiscDistribution();
  subsetSizes->addint(0, leftExamples);
  subsetSizes->addint(1, rightExamples);


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
  PITERATE(vector<int>, mi, bestMapping) {
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



TTreeSplitConstructor_Threshold::TTreeSplitConstructor_Threshold(PMeasureAttribute meas, const float &worst, const float &aml)
: TTreeSplitConstructor_Measure(meas, worst, aml)
{}



PClassifier TTreeSplitConstructor_Threshold::operator()(
                             PStringList &descriptions, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute,

                             PExampleGenerator gen, const int &weightID ,
                             PDomainContingency dcont, PDistribution apriorClass,
                             const vector<bool> &candidates
                            )
{ checkProperty(measure);

  // This we cannot offer: we would need to go through all examples each time and the attribute measure (say Relief) as well. Too slow.
  if (measure->needs==TMeasureAttribute::Generator)
    raiseError("cannot use a measure that require example set");

  measure->checkClassTypeExc(gen->domain->classVar->varType);

  bool cse = candidates.size()==0;
  if (!cse && noCandidates(candidates))
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  int N = gen ? gen->numberOfExamples() : -1;
  if (N<0)
    N = dcont->classes->cases;
  TSimpleRandomGenerator rgen(N);

  vector<bool>::const_iterator ci(candidates.begin()), ce(candidates.end());

  PVariable bvar;
  int newpos;
  TDiscDistribution *dis0, *dis1;
  TContDistribution *con0, *con1;
  prepareBinaryCheat(gen, weightID, bvar, dcont, newpos, dis0, dis1, con0, con1);

  PContingency cont = dcont->at(newpos);
  TDiscDistribution *outerDistribution = cont->outerDistribution.AS(TDiscDistribution);

  int thisAttr = 0, bestAttr = -1, wins = 0;
  quality = 0.0;
  float leftExamples, rightExamples;
  float bestThreshold = 0.0;

  TDomainContingency::iterator dci(dcont->begin()), dce(dcont->end());
  for(; (cse || (ci!=ce)) && (dci!=dce); dci++, thisAttr++)
    if ((cse || *(ci++)) && ((*dci)->outerVariable->varType==TValue::FLOATVAR) && (*dci)->continuous->size()>=2) {
      const TDistributionMap &distr = *(*dci)->continuous;

      outerDistribution->unknowns = (*dci)->outerDistribution->unknowns;
      outerDistribution->cases = (*dci)->outerDistribution->cases;
      outerDistribution->abs = (*dci)->outerDistribution->abs;
      outerDistribution->normalized = (*dci)->outerDistribution->normalized;

      int binWins = 0;
      float binQuality = -1.0;
      float binLeftExamples = -1.0, binRightExamples = -1.0;
      TDistributionMap::const_iterator threshold(distr.begin()), threshe(distr.end()), binBestThreshold;

      if (dis0) { // class is discrete
        *dis0 = TDiscDistribution();
        *dis1 = CAST_TO_DISCDISTRIBUTION((*dci)->innerDistribution);
  
        while ((dis0->abs<minSubset) && (threshold!=threshe)) {
          *dis0 += (*threshold).second;
          *dis1 -= (*threshold).second;
          ++threshold;
        };

        while ((dis1->abs>minSubset) && (threshold!=threshe)) {
          outerDistribution->distribution[0] = dis0->abs;
          outerDistribution->distribution[1] = dis1->abs;

          float thisMeas = measure->operator()(newpos, dcont, apriorClass);
          if (   ((!binWins) || (thisMeas>binQuality)) && ((binWins=1) ==1)
              || (thisMeas==binQuality) && rgen.randbool(++binWins)) {
            binBestThreshold = threshold; 
            binQuality = thisMeas;
            binLeftExamples = dis0->abs;
            binRightExamples = dis1->abs;
          }

          *dis0 += (*threshold).second;
          *dis1 -= (*threshold).second;
          ++threshold;
        };
      }
      else { // class is continuous
        *con0 = TContDistribution();
        *con1 = CAST_TO_CONTDISTRIBUTION((*dci)->innerDistribution);

        do {
          *con0 += (*threshold).second;
          *con1 -= (*threshold).second;
          ++threshold;

          if ((con0->abs<minSubset) || (con1->abs<minSubset))
            continue;

          cont->outerDistribution->setint(0, con0->abs);
          cont->outerDistribution->setint(1, con1->abs);
          
          float thisMeas = measure->operator()(newpos, dcont, apriorClass);
          if (   ((!binWins) || (thisMeas>binQuality)) && ((binWins=1) ==1)
              || (thisMeas==binQuality) && rgen.randbool(++binWins)) {
            binBestThreshold = threshold; 
            binQuality = thisMeas;
            binLeftExamples = con0->abs;
            binRightExamples = con1->abs;
          }
        } while (threshold!=threshe);
      }

      if (       binWins
          && (   (!wins || (binQuality>quality)) && ((wins=1)==1)
              || (binQuality==quality) && rgen.randbool(++wins))) {
        bestAttr = thisAttr;
        quality = binQuality;
        leftExamples = binLeftExamples;
        rightExamples = binRightExamples;

        bestThreshold = (*binBestThreshold).first;
        binBestThreshold--;
        bestThreshold += (*binBestThreshold).first;
        bestThreshold /= 2.0;
      }
    }
  
  dcont->erase(dcont->begin() + newpos); // removes the added attribute from the domain contingency

  if (!wins)
    return returnNothing(descriptions, subsetSizes, quality, spentAttribute);

  if (quality<worstAcceptable)
    return returnNothing(descriptions, subsetSizes, spentAttribute);


  subsetSizes = mlnew TDiscDistribution();
  subsetSizes->addint(0, leftExamples);
  subsetSizes->addint(1, rightExamples);

  descriptions = mlnew TStringList();
  char str[128];
  sprintf(str, "<%3.3f", bestThreshold);
  descriptions->push_back(str);
  sprintf(str, ">=%3.3f", bestThreshold);
  descriptions->push_back(str);

  bvar->name = gen->domain->attributes->at(bestAttr)->name;
  spentAttribute = -1;
  return mlnew TClassifierFromVarFD(bvar, gen->domain, bestAttr, subsetSizes, mlnew TThresholdDiscretizer(bestThreshold));
}



PExampleGeneratorList TTreeExampleSplitter::prepareGeneratorList(int size, PDomain domain, vector<TExamplePointerTable *> &unwrapped)
{
  PExampleGeneratorList examplePtrs = mlnew TExampleGeneratorList();
  while(size--) {
    TExamplePointerTable *ntable = mlnew TExamplePointerTable(domain);
    examplePtrs->push_back(PExampleGenerator(ntable));
    unwrapped.push_back(ntable);
  }

  return examplePtrs;
}


bool TTreeExampleSplitter::getBranchIndices(PTreeNode node, PExampleGenerator gen, vector<int> &indices)
{
  TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
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
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  const int maxIndex = node->branchDescriptions->size();

  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);
  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToCommon::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  const int maxIndex = node->branchDescriptions->size();
  const int mostCommon = node->branchSizes->highestProbIntIndex();

  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    uexamplePtrs[!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex) ? index.intV : mostCommon]->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToAll::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  const int maxIndex = node->branchDescriptions->size();

  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);

  PEITERATE(ei, gen) {
    TValue index = branchSelector(*ei);
    if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex))
      uexamplePtrs[index.intV]->addExample(*ei);
    else
      ITERATE(vector<TExamplePointerTable *>, pei, uexamplePtrs)
        (*pei)->addExample(*ei);
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsToRandom::operator ()(PTreeNode node, PExampleGenerator gen, const int &, vector<int> &)
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  const int maxIndex = node->branchDescriptions->size();

  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);

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
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  int maxIndex = node->branchDescriptions->size();
  node->branchDescriptions->push_back("unknown");

  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex+1, gen->domain, uexamplePtrs);

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
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  int maxIndex = node->branchDescriptions->size();

 
  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);

  vector<int> indices;
  
  if (getBranchIndices(node, gen, indices)) {
    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      ++ei;
    }
  }

  else {
    const TDiscDistribution &branchSizes = node->branchSizes.getReference();
    for(int i = maxIndex; i--; )
      newWeights.push_back(getMetaID());

    TExampleIterator ei(gen->begin());
    ITERATE(vector<int>, ii, indices) {
      uexamplePtrs[*ii]->addExample(*ei);
      (*ei).meta.setValue(newWeights[*ii], TValue(WEIGHT(*ei)));
      ++ei;
    }

    for (; ei; ++ei) {
      TValue index = branchSelector(*ei);

      if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex)) {
        uexamplePtrs[index.intV]->addExample(*ei);
        (*ei).meta.setValue(newWeights[index.intV], TValue(WEIGHT(*ei)));
      }
    
      else {
        if (index.isDC()) {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            uexamplePtrs[branchNo]->addExample(*ei);
            (*ei).meta.setValue(newWeights[branchNo], TValue(WEIGHT(*ei)));
          }
        }
        else {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            float weight = branchSizes.p(branchNo) * WEIGHT(*ei);
            if (weight) {
              uexamplePtrs[branchNo]->addExample(*ei);
              (*ei).meta.setValue(newWeights[branchNo], TValue(weight));
            }
          }
        }
      }
    }
  }

  return examplePtrs;
}


PExampleGeneratorList TTreeExampleSplitter_UnknownsAsSelector::operator()(PTreeNode node, PExampleGenerator gen, const int &weightID, vector<int> &newWeights)
{ TClassifier &branchSelector = const_cast<TClassifier &>(node->branchSelector.getReference());
  int maxIndex = node->branchDescriptions->size();

 
  vector<TExamplePointerTable *> uexamplePtrs;
  PExampleGeneratorList examplePtrs = prepareGeneratorList(maxIndex, gen->domain, uexamplePtrs);

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
      (*ei).meta.setValue(newWeights[*ii], TValue(WEIGHT(*ei)));
      ++ei;
    }

    for (; ei; ++ei) {
      TValue index = branchSelector(*ei);

      if (!index.isSpecial() && (index.intV>=0) && (index.intV<maxIndex)) {
        uexamplePtrs[index.intV]->addExample(*ei);
        (*ei).meta.setValue(newWeights[index.intV], TValue(WEIGHT(*ei)));
      }
    
      else {
        if (index.isDC()) {
          for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
            uexamplePtrs[branchNo]->addExample(*ei);
            (*ei).meta.setValue(newWeights[branchNo], TValue(WEIGHT(*ei)));
          }
        }
        else {
          TDiscDistribution *distr = index.svalV ? index.svalV.AS(TDiscDistribution) : NULL;
          if (distr)
            for(int branchNo = 0; branchNo<maxIndex; branchNo++) {
              float weight = distr->p(branchNo) * WEIGHT(*ei);
              if (weight) {
                uexamplePtrs[branchNo]->addExample(*ei);
                (*ei).meta.setValue(newWeights[branchNo], TValue(weight));
            }
          }
        }
      }
    }
  }

  return examplePtrs;
}
