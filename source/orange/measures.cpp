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
#include <set>

#include "stladdon.hpp"
#include "student.hpp"
#include "random.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "distance.hpp"
#include "contingency.hpp"
#include "classify.hpp"

#include "cost.hpp"
#include <vector>

#include "relief.ppp"
#include "measures.ppp"


void checkDiscrete(const PContingency &cont, char *measure)
{ if (cont->varType!=TValue::INTVAR)
    if (cont->outerVariable)
      raiseErrorWho(measure, "cannot evaluate the non-discrete attribute '%s'", cont->outerVariable->name.c_str());
    else
      raiseErrorWho(measure, "cannot evaluate continuous attributes");

  if (cont->innerVariable) {
    if (cont->innerVariable->varType != TValue::INTVAR)
      raiseErrorWho(measure, "cannot work with continuous outcome '%s'", cont->innerVariable->name.c_str());
  }
  else
    if (!cont->innerDistribution.is_derived_from(TDiscDistribution))
      raiseErrorWho(measure, "expects discrete class attribute");
}


void checkDiscreteContinuous(const PContingency &cont, char *measure)
{ if (cont->varType!=TValue::INTVAR)
    if (cont->outerVariable)
      raiseErrorWho(measure, "cannot evaluate the non-discrete attribute '%s'", cont->outerVariable->name.c_str());
    else
      raiseErrorWho(measure, "cannot evaluate continuous attributes");

  if (cont->innerVariable) {
    if (cont->innerVariable->varType != TValue::FLOATVAR)
      raiseErrorWho(measure, "cannot work with discrete outcome '%s'", cont->innerVariable->name.c_str());
  }
  else
    if (!cont->innerDistribution.is_derived_from(TContDistribution))
      raiseErrorWho(measure, "expects continuous outcome");
}


TMeasureAttribute::TMeasureAttribute(const int &aneeds, const bool &hd, const bool &hc)
: needs(aneeds),
  handlesDiscrete(hd),
  handlesContinuous(hc)
{}


float TMeasureAttribute::operator()(PContingency, PDistribution, PDistribution)
{ raiseError("cannot evaluate attribute from contingencies only"); 
  return 0.0;
}


float TMeasureAttribute::operator()(int attrNo, PDomainContingency domainContingency, PDistribution apriorClass)
{ if (needs>Contingency_Class) 
    raiseError("cannot evaluate attribute from domain contingency only");
  if (attrNo>int(domainContingency->size()))
    raiseError("attribute index out of range");
  return operator()(domainContingency->operator[](attrNo), domainContingency->classes, apriorClass); 
}


float TMeasureAttribute::operator()(int attrNo, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{ 
  _ASSERT(gen && gen->domain);
  if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");
  if (attrNo>int(gen->domain->attributes->size()))
    raiseError("attribute index out of range");

  if (needs==Contingency_Class) {
    TContingencyAttrClass contingency(gen, attrNo, weightID);

    PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
    classDistribution->operator+= (contingency.innerDistributionUnknown);

    return operator()(PContingency(contingency), classDistribution, apriorClass);
  }
   
 if (needs>DomainContingency)
   raiseError("invalid 'needs'");

 TDomainContingency domcont(gen, weightID);
 return operator()(attrNo, PDomainContingency(domcont), apriorClass);
}


float TMeasureAttribute::operator ()(PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID)
{ if (!gen->domain->classVar)
    raiseError("can't evaluate attributes on class-less domains");
  
  int attrNo=gen->domain->getVarNum(var, false);
  if (attrNo != ILLEGAL_INT)
    return operator()(attrNo, gen, apriorClass, weightID);

  if (needs>Contingency_Class)
    raiseError("invalid 'needs'");

  TContingencyAttrClass contingency(gen, var, weightID);

  PDistribution classDistribution = CLONE(TDistribution, contingency.innerDistribution);
  classDistribution->operator+= (contingency.innerDistributionUnknown);

  return operator()(PContingency(contingency), PDistribution(classDistribution), apriorClass);
}


float TMeasureAttribute::operator ()(PDistribution dist) const
{ TDiscDistribution *discdist = dist.AS(TDiscDistribution);
  if (discdist)
    return operator()(*discdist);
  
  TContDistribution *contdist = dist.AS(TContDistribution);
  if (contdist)
    return operator()(*contdist);
    
  raiseError("invalid distribution");
  return 0.0;
}

float TMeasureAttribute::operator ()(const TDiscDistribution &) const
{ raiseError("cannot evaluate discrete attributes");
  return 0.0;
}

float TMeasureAttribute::operator ()(const TContDistribution &) const
{ raiseError("cannot evaluate continuous attributes");
  return 0.0;
}



bool TMeasureAttribute::checkClassType(const int &varType)
{
  return    ((varType==TValue::INTVAR) && handlesDiscrete)
         || ((varType==TValue::FLOATVAR) && handlesContinuous);
}


void TMeasureAttribute::checkClassTypeExc(const int &varType)
{
  if (varType==TValue::INTVAR) {
    if (!handlesDiscrete)
      raiseError("cannot evaluate discrete attributes");
  }
  else if (varType==TValue::FLOATVAR) {
    if (!handlesContinuous)
      raiseError("cannot evaluate continuous attributes");
  }
}


TMeasureAttributeFromProbabilities::TMeasureAttributeFromProbabilities(const bool &hd, const bool &hc, const int &unkTreat)
: TMeasureAttribute(Contingency_Class, hd, hc),
  unknownsTreatment(unkTreat)
{}


float TMeasureAttributeFromProbabilities::operator()(PContingency cont, PDistribution classDistribution, PDistribution aprior)
{ 
  if (estimator) {
    classDistribution = estimator->call(classDistribution, aprior)->call();
    if (!classDistribution)
      raiseError("'estimator' cannot return the distribution");
  }

  if (conditionalEstimator) {
    cont = conditionalEstimator->call(cont, aprior)->call();
    if (!cont)
      raiseError("'conditionalEstimator cannot return contingency matrix");
  }

  return operator()(cont, CAST_TO_DISCDISTRIBUTION(unknownsTreatment == UnknownsToCommon ? classDistribution : cont->innerDistribution));
}


inline float round0(const float &x)
{ 
  return (x > -1e-6) && (x < 1e-6) ? 0.0 : x;
}


float getEntropy(const vector<float> &vf)
{ 
  float n = 0.0, sum = 0.0;
  int noDif0 = 0;
  const_ITERATE(vector<float>, vi, vf)
    if (*vi>0) {
      sum += (*vi)*log(*vi);
      n += *vi;
      noDif0++;
    }
  return (noDif0>1) ? (log(float(n))-sum/n) / log(2.0) : 0;
}


float getEntropy(PContingency cont, bool unknownsToCommon)
{ 
  checkDiscrete(cont, "getEntropy");

  float sum = 0.0, N = 0.0;
  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  TDistributionVector::const_iterator mostCommon = unknownsToCommon ? cont->discrete->begin() + outer.highestProbIntIndex() : cont->discrete->end();
  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += cont->innerDistributionUnknown;
      N += dist2.cases;
      sum += dist2.cases * getEntropy(dist2);
    }
    else {
      N += dist.cases;
      sum += dist.cases * getEntropy(dist.distribution);
    }
  }
  return N ? sum/N : 0.0;
}



TMeasureAttribute_info::TMeasureAttribute_info(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}


float TMeasureAttribute_info::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  // checkDiscrete is called by getEntropy

  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.cases == 0))
    return 0.0;

  float info = getEntropy(classProbabilities) - getEntropy(probabilities, unknownsTreatment==UnknownsToCommon);
  if (unknownsTreatment == ReduceByUnknowns)
    info *= outer.cases / (outer.unknowns + outer.cases);

  return round0(info);
}


float TMeasureAttribute_info::operator()(const TDiscDistribution &dist) const
{ 
  return -getEntropy(dist);
}


TMeasureAttribute_gainRatio::TMeasureAttribute_gainRatio(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}



float TMeasureAttribute_gainRatio::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_gainRatio");

  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(probabilities->outerDistribution);
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;

  float attributeEntropy = getEntropy(outer);
  if (attributeEntropy<1e-20)
    return 0.0;
  
  float gain = getEntropy(classProbabilities) - getEntropy(probabilities, unknownsTreatment==UnknownsToCommon);
  if (gain<1e-20)
    return 0.0;
  
  gain /= attributeEntropy;

  if (unknownsTreatment == ReduceByUnknowns)
    gain *= outer.cases / (outer.unknowns + outer.cases);

  return round0(gain);
}



float TMeasureAttribute_gainRatioA::operator()(const TDiscDistribution &dist) const
{ return round0(-getEntropy(dist) * log(float(dist.size()))); }




float getGini(const vector<float> &vf)
{ float sum = 0.0, N = 0.0;
  const_ITERATE(vector<float>, vi, vf) {
    N += *vi;
    sum += (*vi)*(*vi);
  }
  return N ? 1 - sum/N/N : 0.0;
}


float getGini(PContingency cont, bool unknownsToCommon)
{ 
  float sum = 0.0, N = 0.0;
  const TDiscDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  TDistributionVector::const_iterator mostCommon = unknownsToCommon ? cont->discrete->begin() + outer.highestProbIntIndex() : cont->discrete->end();
  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += cont->innerDistributionUnknown;
      N += dist2.cases;
      sum += dist2.cases * getGini(dist2);
    }
    else {
      N += dist.cases;
      sum += dist.cases * getGini(dist.distribution);
    }
  }
  return N ? sum/N : 0.0;
}


TMeasureAttribute_gini::TMeasureAttribute_gini(const int &unk)
: TMeasureAttributeFromProbabilities(true, false, unk)
{}


float TMeasureAttribute_gini::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_gini");
  
  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;
 
  float gini = getGini(classProbabilities) - getGini(probabilities, unknownsTreatment==UnknownsToCommon);
  if (unknownsTreatment == ReduceByUnknowns)
    gini *= (outer.cases / (outer.unknowns + outer.cases));

  return round0(gini);
}


float TMeasureAttribute_gini::operator()(const TDiscDistribution &dist) const
{ return round0(-getGini(dist)); }



TMeasureAttribute_cost::TMeasureAttribute_cost(PCostMatrix costs)
: TMeasureAttributeFromProbabilities(true, false),
  cost(costs)
{}


float TMeasureAttribute_cost::majorityCost(const TDiscDistribution &dval)
{ float cost;
  TValue cclass;
  majorityCost(dval, cost, cclass);
  return cost;
}


void TMeasureAttribute_cost::majorityCost(const TDiscDistribution &dval, float &ccost, TValue &cclass)
{ 
  checkProperty(cost);

  int dsize = dval.size();
  if (dsize>cost->size())
    raiseError("cost matrix is too small");

  TRandomGenerator srgen(dval.sumValues());

  ccost = numeric_limits<float>::max();
  int wins = 0, bestPrediction;

  for(int predicted = 0; predicted < dsize; predicted++) {
    float thisCost = 0;
    for(int correct = 0; correct < dsize; correct++)
      thisCost += dval[correct] * cost->getCost(predicted, correct);

    if (   (thisCost<ccost) && ((wins=1)==1)
        || (thisCost==ccost) && srgen.randbool(++wins)) {
      bestPrediction = predicted;
      ccost = thisCost; 
    }
  }
  
  ccost /= dval.abs;
  cclass = TValue(bestPrediction);
}


float TMeasureAttribute_cost::operator()(PContingency probabilities, const TDiscDistribution &classProbabilities)
{ 
  checkDiscrete(probabilities, "MeasureAttribute_cost");

  const TDistribution &outer = probabilities->outerDistribution.getReference();
  if ((unknownsTreatment == ReduceByUnknowns) && (outer.unknowns == outer.cases))
    return 0.0;
 
  checkProperty(cost);

  float stopCost = majorityCost(classProbabilities);
  
  TDistributionVector::const_iterator mostCommon = (unknownsTreatment == UnknownsToCommon)
    ? probabilities->discrete->begin() + outer.highestProbIntIndex()
    : probabilities->discrete->end();

  float continueCost = 0;
  float N = 0;
  const_ITERATE(TDistributionVector, ci, *probabilities->discrete) {
    const TDiscDistribution &dist = CAST_TO_DISCDISTRIBUTION(*ci);
    if (ci == mostCommon) {
      TDiscDistribution dist2 = dist;
      dist2 += probabilities->innerDistributionUnknown;
      N += dist2.cases;
      continueCost += dist2.cases * majorityCost(dist2);
    }
    else {
      N += dist.cases;
      continueCost += dist.cases * majorityCost(dist.distribution);
    }
  }
  continueCost /= N;

  float cost = stopCost - continueCost;
  if (unknownsTreatment == ReduceByUnknowns)
    cost *= (outer.cases / (outer.unknowns + outer.cases));

  return round0(cost);
}



TMeasureAttribute_MSE::TMeasureAttribute_MSE()
: TMeasureAttribute(Contingency_Class, false, true),
  m(0)
{}


float TMeasureAttribute_MSE::operator()(PContingency cont, PDistribution classDistribution, PDistribution apriorClass)
{
  checkDiscreteContinuous(cont, "MeasureAttribute_MSE");

  const TDistribution &outer = CAST_TO_DISCDISTRIBUTION(cont->outerDistribution);
  
  if (cont->innerVariable->varType!=TValue::FLOATVAR)
    raiseError("cannot evaluate attribute in domain with discrete classes");
  if (cont->outerVariable->varType!=TValue::INTVAR)
    raiseError("cannot evaluate continuous attributes");

  const TContDistribution &classDist = CAST_TO_CONTDISTRIBUTION(classDistribution);

  float W=classDist.abs;
  if (W<=0)
    return 0.0;

  float I_orig=(classDist.sum2-classDist.sum*classDist.sum/W)/W;
  if (I_orig<=0.0)
    return 0.0;

  TDistributionVector::const_iterator mostCommon = (unknownsTreatment == UnknownsToCommon)
    ? cont->discrete->begin() + outer.highestProbIntIndex()
    : cont->discrete->end();

  float I=0;
  float downW=0;
  const_ITERATE(TDistributionVector, ci, *cont->discrete) {
    const TContDistribution &tdist = CAST_TO_CONTDISTRIBUTION(*ci);
    if (ci==mostCommon) {
      const float ssum2 = tdist.sum2 + cont->innerDistribution.AS(TContDistribution)->sum2;
      const float ssum = tdist.sum + cont->innerDistribution.AS(TContDistribution)->sum;
      const float sabs = tdist.abs + cont->innerDistribution.AS(TContDistribution)->abs;
      I += ssum2  -  ssum*ssum / sabs;
      downW += sabs;
    }
    else {
      if (tdist.abs>0) {
        I += tdist.sum2 - tdist.sum*tdist.sum/tdist.abs;
       downW += tdist.abs;
      }
    }
  }

  if (apriorClass && (m>0)) {
    const TContDistribution &tdist = CAST_TO_CONTDISTRIBUTION(apriorClass);
    I =   (I + m * (tdist.sum2 - tdist.sum * tdist.sum/tdist.abs) / tdist.abs)
        / (downW + m);
  }
  else 
    I /= downW;

  float mse = (I_orig - I)/I_orig;
  if (unknownsTreatment == ReduceByUnknowns)
    mse *= (outer.cases / (outer.unknowns + outer.cases));
  
  return round0(mse);
}



TMeasureAttribute_relief::TMeasureAttribute_relief(int ak, int am)
: TMeasureAttribute(Generator, true, false), 
  k(ak),
  m(am)
{}


class TDistRec {
public:
  float dist;
  long randoff;
  TExample *example;

  TDistRec(TExample *anex, const int &roff, float adist)
    : dist(adist),
      randoff(roff),
      example(anex)
    {};

  bool operator <(const TDistRec &other) const
    { return (dist==other.dist) ? (randoff<other.randoff) : (dist<other.dist); }
  bool operator !=(const TDistRec &other) const
    { return    (dist!=other.dist) || (randoff!=other.randoff); }
};



float TMeasureAttribute_relief::operator()(int attrNo, PExampleGenerator gen, PDistribution aprior, int weightID)
{
  if (!gen->domain->classVar)
    raiseError("class-less domain");

/* If the generator's address, domain, number of examples and weight are same as before,
   it concludes that the data is the same... (this is a bit dangerous, though) */
  if (   (gen!=prevGenerator)
      || (gen->domain!=prevDomain)
      || (gen->domain->version!=prevDomainVersion)
      || (weightID!=prevWeight)
      || (gen->numberOfExamples()!=prevGenerator->numberOfExamples())) {

    vector<TExampleTable *> tables;
    TExampleTable *examples = NULL;

    TRandomGenerator rgen(gen->numberOfExamples());

    try {
      PExamplesDistance wdistance = TExamplesDistanceConstructor_Relief()(gen);
      const TExamplesDistance_Relief &distance = dynamic_cast<const TExamplesDistance_Relief &>(wdistance.getReference());
  
      if (gen->domain->classVar->varType==TValue::INTVAR) {
        // prepares tables of examples of different classes
        for (int i = gen->domain->classVar->noOfValues(); i--; )
          // if gen is ExampleTable, our tables won't own examples (ie don't need to copy them)
          tables.push_back(mlnew TExampleTable(gen->domain, !gen.is_derived_from(TExampleTable)));

        PEITERATE(ei, gen)
          if (!(*ei).getClass().isSpecial())
            (tables.at(int((*ei).getClass())))->addExample(*ei);

        // the total number of examples and number of examples of each class
        long N=0;
        vector<long> gN;
        { ITERATE(vector<TExampleTable *>, gi, tables) {
            gN.push_back((*gi)->numberOfExamples());
            N+=gN.back();
          }
        }

        if (!N)
          raiseError("no examples with known class");

        measures = vector<float>(gen->domain->attributes->size(), 0);
        // This is what the measures must be divided by - the sum of products of weights of compared examples
        // Should be m*k but can be less if some class doesn't have enough examples or
        // a bit more if referenceExamples exceeds m.
        float actualN=0;

        for(float referenceExamples=0, refWeight; referenceExamples<m; referenceExamples+=refWeight) {
          // choose a random example
          long eNum = rgen.randlong(N);
          int eClass=0;
          for(; eNum>=gN[eClass]; eNum-=gN[eClass++]);
          TExample &example = *tables[eClass]->examples[eNum];
          refWeight=WEIGHT(example);

          // for each class
          int tsize = tables.size();
          for(int oClass=0; oClass<tsize; oClass++) 
            if (tables[oClass]->numberOfExamples()>0) {
              // sort the examples by the distance
              set<TDistRec> neighset;
              EITERATE(ei, *tables[oClass])
                neighset.insert(TDistRec(&*ei, rgen.randlong(), distance(example, *ei)));

              float classWeight= (oClass==eClass) ? -1.0 : float(gN[oClass]) / float(N-gN[eClass]);

              set<TDistRec>::iterator ni(neighset.begin()), ne(neighset.end());
              while(((*ni).dist<=0) && (ni!=ne))
                ni++;

              for(float needwei=k, compWeight; (needwei>0) && (ni!=ne); needwei-=compWeight, ni++) {
                // determine the weight of the current example; weights are negative for same classes
                compWeight=WEIGHT(*(*ni).example);
                if (compWeight>needwei) compWeight=needwei;
                float koe=refWeight*compWeight*classWeight;
                actualN+=fabs(koe);

                // add the (weighted) distance
                int attrNo=0;
                TExample::iterator e1i(example.begin()), e2i((*ni).example->begin());
                for(vector<float>::iterator mi(measures.begin()), me(measures.end());
                    mi!=me; 
                    *(mi++) += koe * distance(attrNo++, *(e1i++), *(e2i++)));
              }
            }
        }

        ITERATE(vector<TExampleTable *>, gi, tables) {
          mldelete *gi;
          *gi = NULL;
        }
        tables.clear();

        ITERATE(vector<float>, mi, measures)
          *mi/=actualN;
      }

      else {

        TExampleIterator test1(gen->begin());
        if (!test1)
          measures = vector<float>(gen->domain->attributes->size(), 0);
 
        else {
          measures.clear();
          examples = mlnew TExampleTable(gen, !gen.is_derived_from(TExampleTable)); // This will automatically store examples into ExampleTable if necessary

          float minCl, maxCl;
          { TExampleIterator emi(examples->begin());
            minCl=maxCl=(*emi).getClass().floatV;
            while(++emi) {
              float tex=(*emi).getClass().floatV;
	            if (tex>maxCl) maxCl=tex;
		          else if (tex<minCl) minCl=tex;
		        }
          }

          float NdC=0;
          float actualN=0;
          vector<float> NdA(gen->domain->attributes->size(), 0);
          vector<float> NdCdA(gen->domain->attributes->size(), 0);

          for(float referenceExamples=0, refWeight; referenceExamples<m; referenceExamples+=refWeight) {
            // choose a random example
            long eNum = rgen.randlong(examples->numberOfExamples());
            TExample &refExample = *examples->examples[eNum];
            refWeight=WEIGHT(refExample);
            float refClass=refExample.getClass().floatV;

            // for each class
            // sort the examples by the distance
            set<TDistRec> neighset;
            EITERATE(ei, *examples)
              neighset.insert(TDistRec(&*ei, rgen.randlong(), distance(refExample, *ei)));

            set<TDistRec>::iterator ni(neighset.begin()), ne(neighset.end());
            while(((*ni).dist<=0) && (ni!=ne)) ni++;
            for(float needwei=k, compWeight; (needwei>0) && (ni!=ne); needwei-=compWeight, ni++) {
              TExample &compExample=*(*ni).example;
              compWeight=WEIGHT(compExample);
              if (compWeight>needwei) compWeight=needwei;
              float koe=refWeight*compWeight;
              float classDiff=fabs(compExample.getClass().floatV-refClass);
              actualN+=koe;

              NdC+=koe*classDiff;
              // add the (weighted) distance
              int attrNo=0;
              TExample::iterator e1i(refExample.begin()), e2i(compExample.begin());
              for(vector<float>::iterator dAi(NdA.begin()), dCdAi(NdCdA.begin()), dAe(NdA.end());
                  dAi!=dAe;
			           ) {
                float diff=koe * distance(attrNo++, *(e1i++), *(e2i++));
                *(dAi++)+=diff;
                *(dCdAi++)+=classDiff*diff*koe;
              }
		        }
          }

          NdC=(NdC-actualN*minCl)/(maxCl-minCl);
          for(vector<float>::iterator dAi(NdA.begin()), dCdAi(NdCdA.begin()), dAe(NdA.end()); dAi!=dAe; dAi++, dCdAi++)
	        measures.push_back(*dCdAi / NdC - (*dAi - *dCdAi)/(actualN-NdC));
          ITERATE(vector<float>, mi, measures)
            *mi/=actualN;

          mldelete examples;
        }
      }

      prevDomain=gen->domain;
      prevDomainVersion=prevDomain->version;
      prevGenerator=gen;
      prevExamples=gen->numberOfExamples();
      prevWeight=weightID;
    }
    catch (exception err) {
      ITERATE(vector<TExampleTable *>, gi, tables)
        delete *gi;
      if (examples)
        delete examples;
      throw;
    }
  }


  float me=measures[attrNo];
  return me;
}
