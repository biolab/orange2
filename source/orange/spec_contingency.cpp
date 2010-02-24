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
#include "learn.hpp"
#include "classify.hpp"
#include "preprocessors.hpp"

#include "spec_contingency.ppp"

PDomainContingency TComputeDomainContingency_DomainTransformation::operator()(PExampleGenerator eg, const long &weightID)
{ checkProperty(domainTransformerConstructor);

  // We'll allow a domainTransformerConstructor that does not generally preserve the attributes.
  // Maybe the user knows that it will preserve the attributes this time.
  // Or he may be just pushing his luck.

  PDomain domain=domainTransformerConstructor->operator()(eg, weightID);
  if (!domain)
    raiseError("'domainTransformerConstructor' did not return a valid domain");

  if (resultInOriginalDomain && !domainTransformerConstructor->preservesOrder) {
    TVarList orderedList;
    for(TVarList::iterator bi(domain->attributes->begin()), be(domain->attributes->end()); bi!=be; bi++) {
      TVarList::iterator si(domain->attributes->begin());
      while((si!=be) && (*si!=*bi) && ((*si)->sourceVariable!=*bi))
        si++;
      if (si==be)
        raiseError("the transformed domain misses the attribute '%s'", (*bi)->name.c_str());
      orderedList.push_back(*si);
    }
    PDomain domain=mlnew TDomain(domain->classVar, orderedList);
  }

  PDomainContingency domainContingency;
  domainContingency->computeMatrix(eg, weightID, NULL, domain);

  if (resultInOriginalDomain) {
    TVarList::iterator oi(eg->domain->attributes->begin());
    PITERATE(TDomainContingency, dci, domainContingency)
      (*dci)->outerVariable=*oi;
  }

  return domainContingency;
}


PDomainContingency TComputeDomainContingency_ImputeWithClassifier::operator ()(PExampleGenerator egen, const long &weightID)
{ 
  PDomain myDomain = egen->domain;  
  PVariable classVar = myDomain->classVar;

  vector<PClassifier> classifiers(myDomain->attributes->size(), PClassifier());
  vector<PDomain> domains(myDomain->attributes->size(), PDomain());

  PDomainDistributions distributions;
  PDomainContingency pureContingencies;

  if (!classVar)
   raiseError("classless domain");

  TDomainContingency *udcont = mlnew TDomainContingency();
  PDomainContingency dcont = PDomainContingency(udcont);

  udcont->classes = TDistribution::create(classVar);
  char classType = classVar->varType;

  // Prepare a TContingency for each attribute that has discrete or continuous values

  TValue lastClassValue;
  if (classVar->varType==TValue::INTVAR)
    lastClassValue = TValue(classVar->noOfValues()-1);

  PITERATE(TVarList, vli, myDomain->attributes) {
    if (classIsOuter)
      udcont->push_back(mlnew TContingencyClassAttr(*vli, myDomain->classVar));
    else
      udcont->push_back(mlnew TContingencyAttrClass(*vli, myDomain->classVar));

    if (((*vli)->varType==TValue::INTVAR) && (classType==TValue::INTVAR))
      for(int i=0, e=(*vli)->noOfValues(); i!=e; i++)
        udcont->back()->add_attrclass(TValue(i), lastClassValue, 0);
  }

  int Na=myDomain->attributes->size();

  PEITERATE(fi, egen) {
    TExample::iterator vi((*fi).begin());
    TExample::iterator cli(vi+Na);
    vector<PClassifier>::iterator ci(classifiers.begin());
    vector<PDomain>::iterator di(domains.begin());
    TDomainContingency::iterator si(udcont->begin());

    float xmplWeight = WEIGHT(*fi);
    udcont->classes->add(*cli, xmplWeight);

    for(; vi!=cli; vi++, si++, ci++, di++)
      if (   (*vi).isSpecial()
          && (*ci || ((*vi).varType==TValue::INTVAR) || ((*vi).varType==TValue::FLOATVAR))) {
        if (!*ci) {
          PLearner learner = (*vi).varType==TValue::INTVAR ? learnerForDiscrete : learnerForContinuous;
          int pos = vi-(*fi).begin();
          switch (learner->needs) {
            case TLearner::NeedsNothing:
              *ci = learner->call(myDomain->attributes->at(pos));
              break;
            case TLearner::NeedsClassDistribution:
              if (!distributions)
                distributions = mlnew TDomainDistributions(egen, weightID);
              *ci = learner->call(distributions->at(pos));
              break;
            case TLearner::NeedsDomainContingency:
              if (!pureContingencies)
                pureContingencies = mlnew TDomainContingency(egen, weightID);
              *ci = learner->call(pureContingencies);
            default:
              vector<PDomain>::iterator di = domains.begin() + (vi-(*fi).begin());
              PVariable vari = myDomain->attributes->at(pos);

              *di = CLONE(TDomain, myDomain);
              (*di)->delVariable(vari);
              (*di)->changeClass(vari);
              (*di)->addVariable(myDomain->classVar);
              
              *ci = learner->call(egen, weightID);
          }
        }
        (*si)->add_attrclass((*ci)->call(TExample(*di, *fi)), *cli, xmplWeight);
      }
      else
        (*si)->add_attrclass(*vi, *cli, xmplWeight);
  }

  return dcont;
}
  


PDomainContingency TComputeDomainContingency_Preprocessor::operator()(PExampleGenerator egen, const long &weightID)
{ checkProperty(preprocessor);

  
  int newWeight;
  PExampleGenerator newGen=preprocessor->call(egen, weightID, newWeight);
  if (!newWeight)
    newWeight = weightID;

  PDomainContingency domainContingency;
  domainContingency->computeMatrix(newGen, newWeight);
    
  if (resultInOriginalDomain) { // if everything's OK, we'll only need to do some shuffling here
    for(TDomainContingency::iterator dci(domainContingency->begin()), dce(domainContingency->end()); dci!=dce; dci++) {
      TDomainContingency::iterator sci(dci);
      while((dci!=dce) && ((*sci)->outerVariable!=(*dci)->outerVariable) && ((*sci)->outerVariable->sourceVariable!=(*dci)->outerVariable))
        sci++;
      if (dci==dce)
        raiseError("preprocessed examples miss the attribute '%s'", (*dci)->outerVariable->name.c_str());
      else if ((*sci)->outerVariable->sourceVariable==(*dci)->outerVariable) {
        (*sci)->outerVariable = (*dci)->outerVariable;
        PDomainContingency tempc = *dci;
        *dci = *sci;
        *sci = tempc;
      }
    }
  }

  return domainContingency;
}
