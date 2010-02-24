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


#include "stladdon.hpp"

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "classify.hpp"
#include "estimateprob.hpp"

#include "learn.hpp"

#include "contingency.ppp"

DEFINE_TOrangeVector_classDescription(PContingencyClass, "TContingencyClassList", true, ORANGE_API)

#define NOTSPEC(v) if (v.isSpecial()) throw mlexception("unknown variable value");
#define NEEDS(ptype) if(varType!=ptype) throw mlexception("invalid variable type");


// Initializes type field and discrete/continuous field, whichever appropriate.
TContingency::TContingency(PVariable var, PVariable innervar)
: outerVariable(var),
  innerVariable(innervar),
  varType(var ? var->varType : TValue::NONE),
  outerDistribution(TDistribution::create(var)),
  discrete((TDistributionVector *)NULL),
  innerDistribution(TDistribution::create(innervar)),
  innerDistributionUnknown(TDistribution::create(innervar))
{ 
  if (varType==TValue::INTVAR) {
    discrete = mlnew TDistributionVector();
    for(int i=0, e=outerVariable->noOfValues(); i!=e; i++)
      discrete->push_back(TDistribution::create(innervar));
  }
  else if (varType==TValue::FLOATVAR)
    continuous = mlnew TDistributionMap();
}


TContingency::TContingency(const TContingency &old)
: outerVariable(old.outerVariable),
  innerVariable(old.innerVariable),
  varType(old.varType),
  discrete((TDistributionVector *)NULL),
  outerDistribution(CLONE(TDistribution, old.outerDistribution)),
  innerDistribution(CLONE(TDistribution, old.innerDistribution)),
  innerDistributionUnknown(CLONE(TDistribution, old.innerDistributionUnknown))
{ if (varType==TValue::INTVAR)
    discrete = mlnew TDistributionVector(*old.discrete);
  else if (varType==TValue::FLOATVAR)
    continuous = mlnew TDistributionMap(*old.continuous);
}


int TContingency::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);

  if (varType==TValue::INTVAR) {
    PITERATE(TDistributionVector, di, discrete)
      PVISIT(*di);
  }
  else if (varType==TValue::FLOATVAR) {
    PITERATE(TDistributionMap, di, continuous)
      PVISIT((*di).second);
  }

  return 0;
}

int TContingency::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);

  if (varType==TValue::INTVAR)
    mldelete discrete;
  else if (varType==TValue::FLOATVAR)
    mldelete continuous;

  return 0;
}


TContingency &TContingency::operator =(const TContingency &old)
{ outerVariable = old.outerVariable;
  innerVariable = old.innerVariable;
  varType = old.varType;
  innerDistribution = CLONE(TDistribution, old.innerDistribution);
  outerDistribution = CLONE(TDistribution, old.outerDistribution);
  innerDistributionUnknown = CLONE(TDistribution, old.innerDistributionUnknown);

  if (varType==TValue::INTVAR)
    discrete=mlnew TDistributionVector(*old.discrete);
  else if (varType==TValue::FLOATVAR)
    continuous=mlnew TDistributionMap(*old.continuous);
  else discrete=NULL;

  return *this;
}


TContingency::~TContingency()
{ if (varType==TValue::INTVAR)
    mldelete discrete;
  else if (varType==TValue::FLOATVAR)
    mldelete continuous;
}


PDistribution TContingency::operator [](const int &i)                
{ NEEDS(TValue::INTVAR); 
  while (int(discrete->size())<=i) {
    discrete->push_back(TDistribution::create(innerVariable));
    if (innerVariable->varType==TValue::INTVAR)
      discrete->back()->addint(innerVariable->noOfValues()-1, 0);
  }
  return (*discrete)[i];
}


const PDistribution TContingency::operator [](const int &i) const
{ NEEDS(TValue::INTVAR);
  if (!discrete->size())
    raiseError("empty contingency");
  if (i>=int(discrete->size()))
    raiseError("index %i is out of range 0-%i", i, discrete->size()-1);
  return (*discrete)[i];
}


PDistribution TContingency::operator [](const float &i)
{ NEEDS(TValue::FLOATVAR); 
  TDistributionMap::iterator mi=continuous->find(i);
  if (mi==continuous->end()) {
    PDistribution ret = (*continuous)[i] = TDistribution::create(innerVariable);
    if (innerVariable->varType==TValue::INTVAR)
      ret->addint(innerVariable->noOfValues()-1, 0);
    return ret;
  }
  else
    return (*mi).second;
}

    
const PDistribution TContingency::operator [](const float &i) const
{ NEEDS(TValue::FLOATVAR); 
  TDistributionMap::iterator mi = continuous->find(float(i));
  if (mi==continuous->end())
    raiseError("index out of range.");
  return (*mi).second;
}


PDistribution TContingency::operator [](const TValue &i)             
{ NOTSPEC(i);
  return (varType==TValue::INTVAR) ? operator[](int(i)) : operator[](float(i));
}


PDistribution const TContingency::operator [](const TValue &i) const // same, but calls 'const' version of operators[]
{ NOTSPEC(i);
  return (varType==TValue::INTVAR) ? operator[](int(i)) : operator[](float(i)); 
}


PDistribution TContingency::operator [](const string &i)             
{ TValue val;
  checkProperty(outerVariable);
  outerVariable->str2val(i, val);
  return operator[](val);
}


PDistribution const TContingency::operator [](const string &i) const // same, but calls 'const' version of operators[]
{ TValue val;
  checkProperty(outerVariable);
  outerVariable.getReference().str2val(i, val);
  return operator[](val);
}


void TContingency::add(const TValue &outvalue, const TValue &invalue, const float p)
{
  outerDistribution->add(outvalue, p);

  if (outvalue.isSpecial()) {
    innerDistributionUnknown->add(invalue, p);
  }
  else {
    innerDistribution->add(invalue, p);

    switch(outvalue.varType) {
      case TValue::INTVAR:
        if (!outvalue.svalV) {
          (*this)[outvalue]->add(invalue, p);
          return;
        }
        else {
          const TDiscDistribution &dv=dynamic_cast<const TDiscDistribution &>(outvalue.svalV.getReference());
          int i=0;
          float dp=p/dv.abs;
          const_ITERATE(TDiscDistribution, vi, dv)
            (*this)[i++]->add(invalue, dp*(*vi));
          return;
        }

      case TValue::FLOATVAR:
        if (!outvalue.svalV) {
          (*this)[outvalue]->add(invalue, p);
          return;
        }
        else {
          const TContDistribution &dv=dynamic_cast<const TContDistribution &>(outvalue.svalV.getReference());
          float dp=p/dv.abs;
          const_ITERATE(TContDistribution, vi, dv)
            (*this)[(*vi).first]->add(invalue, dp*(*vi).second);
          return;
        }
      default:
        raiseError("unknown value type");
    }
  }
}

       
PDistribution TContingency::p(const int &i) const
{ return operator[](i); }


PDistribution TContingency::p(const string &s) const
{ return operator[](s); }


PDistribution TContingency::p(const TValue &val) const
{ NOTSPEC(val);
  return (varType==TValue::INTVAR) ? p(int(val)) : p(float(val));
}


PDistribution TContingency::p(const float &f) const
{
  NEEDS(TValue::FLOATVAR);
  TDistributionMap::const_iterator i1;
  
  i1 = continuous->end();
  if (f > (*--i1).first)
    return CLONE(TDistribution, (*i1).second);
    
  i1=continuous->lower_bound(f);
  if (i1==continuous->end())
    if (continuous->size()==0)
      raiseError("empty contingency");
    else
      return CLONE(TDistribution, (*(--i1)).second);
  else if (((*i1).first == f) || (i1==continuous->begin()))
    return CLONE(TDistribution, (*i1).second);

  TDistributionMap::const_iterator i2 = i1;
  i1--;

  const float &x1 = (*i1).first;
  const float &x2 = (*i2).first;
  const PDistribution &y1 = (*i1).second;
  const PDistribution &y2 = (*i2).second;

  const float r = (x1==x2) ? 0.5 : (f-x1)/(x2-x1);

  // We want to compute y1*(1-r) + y2*r
  // We know that r!=0, so we can compute (y1*(1-r)/r + y2) * r
  TDistribution *res = CLONE(TDistribution, y1);
  PDistribution wres = res;
  *res *= (1-r)/r;
  *res += y2;
  *res *= r;

  return wres;
}


void TContingency::normalize()
{ if (varType==TValue::INTVAR)
    ITERATE(TDistributionVector, ci, *discrete)
      (*ci)->normalize();
  else if (varType==TValue::FLOATVAR)
    ITERATE(TDistributionMap, ci, *continuous)
      (*ci).second->normalize();
}




TContingencyClass::TContingencyClass(PVariable outer, PVariable inner)
: TContingency(outer, inner)
{}


float TContingencyClass::p_attr(const TValue &, const TValue &) const
{ raiseError("cannot compute p(value|class)"); 
  return 0.0;
}


float TContingencyClass::p_class(const TValue &, const TValue &) const
{ raiseError("cannot compute p(class|value)");
  return 0.0;
}

PDistribution TContingencyClass::p_attrs(const TValue &) const
{ raiseError("cannot compute p(.|class)"); 
  return PDistribution();
}


PDistribution TContingencyClass::p_classes(const TValue &) const
{ raiseError("cannot compute p(class|.)");
  return PDistribution();
}


void TContingencyClass::constructFromGenerator(PVariable outer, PVariable inner, PExampleGenerator gen, const long &weightID, const int &attrNo)
{
  outerVariable = outer;
  innerVariable = inner;

  outerDistribution = TDistribution::create(outerVariable);
  innerDistribution = TDistribution::create(innerVariable);
  innerDistributionUnknown = TDistribution::create(innerVariable);

  varType = outerVariable->varType;
  if (varType==TValue::INTVAR) {
    discrete = mlnew TDistributionVector();
    for(int i=0, e=outerVariable->noOfValues(); i!=e; i++)
      discrete->push_back(TDistribution::create(innerVariable));
  } 
  else {
   _ASSERT(varType==TValue::FLOATVAR);
    continuous = mlnew TDistributionMap();
  }

  if (attrNo == ILLEGAL_INT)
    add_gen(gen, weightID);
  else
    add_gen(gen, attrNo, weightID);
}




TContingencyClassAttr::TContingencyClassAttr(PVariable attrVar, PVariable classVar)
: TContingencyClass(classVar, attrVar)
{}


TContingencyClassAttr::TContingencyClassAttr(PExampleGenerator gen, const int &attrNo, const long &weightID)
{
  const TDomain &domain = gen->domain.getReference();

  if (!domain.classVar)
    raiseError("classless domain");
  if (attrNo>=int(domain.attributes->size()))
    raiseError("attribute index %i out of range", attrNo, domain.attributes->size()-1);

  PVariable attribute = domain.getVar(attrNo, false);
  if (!attribute)
    raiseError("attribute not found");

  constructFromGenerator(domain.classVar, attribute, gen, weightID, attrNo);
}    


TContingencyClassAttr::TContingencyClassAttr(PExampleGenerator gen, PVariable var, const long &weightID)
{ 
  if (!gen->domain->classVar)
    raiseError("classless domain");

  const int attrNo = gen->domain->getVarNum(var, false);
  constructFromGenerator(gen->domain->classVar, var, gen, weightID, attrNo);
}


PVariable TContingencyClassAttr::getClassVar()
{ return outerVariable; }


PVariable TContingencyClassAttr::getAttribute()
{ return innerVariable; }


void TContingencyClassAttr::add_gen(PExampleGenerator gen, const long &weightID)
{ checkProperty(innerVariable);
  int attrNo = gen->domain->getVarNum(innerVariable, false);
  if (attrNo != ILLEGAL_INT)
    PEITERATE(ei, gen)
      add((*ei).getClass(), (*ei)[attrNo], WEIGHT(*ei));
  else {
    if (!innerVariable->getValueFrom)
      raiseError("attribute '%s' is not in the domain and its 'getValueFrom' is not defined", innerVariable->name.c_str());

    TVariable &vfe = innerVariable.getReference();
    PEITERATE(ei, gen)
      add((*ei).getClass(), vfe.computeValue(*ei), WEIGHT(*ei));
  }
}


void TContingencyClassAttr::add_gen(PExampleGenerator gen, const int &attrNo, const long &weightID)
{ PEITERATE(ei, gen)
    add((*ei).getClass(), (*ei)[attrNo], WEIGHT(*ei));
}


void TContingencyClassAttr::add_attrclass(const TValue &varValue, const TValue &classValue, const float &p)
{ add(classValue, varValue, p); }


float TContingencyClassAttr::p_attr(const TValue &varValue, const TValue &classValue) const
{ return p(classValue)->p(varValue); }


PDistribution TContingencyClassAttr::p_attrs(const TValue &classValue) const
{ return p(classValue); }




TContingencyAttrClass::TContingencyAttrClass(PVariable attrVar, PVariable classVar)
: TContingencyClass(attrVar, classVar)
{}


TContingencyAttrClass::TContingencyAttrClass(PExampleGenerator gen, PVariable var, const long &weightID)
{ 
  if (!gen->domain->classVar)
    raiseError("classless domain");

  const int attrNo = gen->domain->getVarNum(var, false);
  constructFromGenerator(var, gen->domain->classVar, gen, weightID, attrNo);
}


TContingencyAttrClass::TContingencyAttrClass(PExampleGenerator gen, const int &attrNo, const long &weightID)
{
  const TDomain &domain = gen->domain.getReference();

  if (!domain.classVar)
    raiseError("classless domain");
  if (attrNo>=int(gen->domain->attributes->size()))
    raiseError("attribute index %i out of range", attrNo, gen->domain->attributes->size()-1);

  PVariable attribute = domain.getVar(attrNo);
  if (!attribute)
    raiseError("attribute not found");

  constructFromGenerator(attribute, domain.classVar, gen, weightID, attrNo);
}


PVariable TContingencyAttrClass::getClassVar()
{ return innerVariable; }


PVariable TContingencyAttrClass::getAttribute()
{ return outerVariable; }


void TContingencyAttrClass::add_gen(PExampleGenerator gen, const long &weightID)
{ int attrNo = gen->domain->getVarNum(outerVariable, false);
  if (attrNo != ILLEGAL_INT)
    PEITERATE(ei, gen)
      add((*ei)[attrNo], (*ei).getClass(), WEIGHT(*ei));
  else {
    if (!outerVariable->getValueFrom)
      raiseError("attribute '%s' is not in the domain and its value cannot be computed", outerVariable->name.c_str());

    TVariable &vfe = outerVariable.getReference();
    PEITERATE(ei, gen)
      add(vfe.computeValue(*ei), (*ei).getClass(), WEIGHT(*ei));
  }
}


void TContingencyAttrClass::add_gen(PExampleGenerator gen, const int &attrNo, const long &weightID)
{ PEITERATE(ei, gen)
    add((*ei)[attrNo], (*ei).getClass(), WEIGHT(*ei));
}


void TContingencyAttrClass::add_attrclass(const TValue &varValue, const TValue &classValue, const float &p)
{ add(varValue, classValue, p); }


float TContingencyAttrClass::p_class(const TValue &varValue, const TValue &classValue) const
{ try {
    return p(varValue)->p(classValue); 
  }
  catch (mlexception exc) {
    // !!! This is extremely ugly. Correct it by asking p not to raise exceptions!
    if (!strcmp(exc.what(), "TDistribution: index out of range."))
      return 0.0;
    else
      throw;
  }
}

PDistribution TContingencyAttrClass::p_classes(const TValue &varValue) const
{ return p(varValue); }




TContingencyAttrAttr::TContingencyAttrAttr(PVariable variable, PVariable innervar)
: TContingency(variable, innervar)
{}


TContingencyAttrAttr::TContingencyAttrAttr(PVariable variable, PVariable innervar, PExampleGenerator gen, const long weightID)
: TContingency(variable, innervar)
{ if (gen)
    operator()(gen, weightID);
}


TContingencyAttrAttr::TContingencyAttrAttr(const int &var, const int &innervar, PExampleGenerator gen, const long weightID)
 : TContingency(gen->domain->getVar(var), gen->domain->getVar(innervar))
{ operator()(gen, weightID); }


void TContingencyAttrAttr::operator()(PExampleGenerator gen, const long weightID)
{ int var=gen->domain->getVarNum(outerVariable, false);
  int invar=gen->domain->getVarNum(innerVariable, false);

  if (var == ILLEGAL_INT)
    if (invar == ILLEGAL_INT)
      PEITERATE(ei, gen) {
        TValue val = outerVariable->computeValue(*ei);
        add(val, innerVariable->computeValue(*ei), WEIGHT(*ei));
      }
    else // var == ILLEGAL_INT, invar is not
      PEITERATE(ei, gen) {
        TValue val = outerVariable->computeValue(*ei);
        add(val, (*ei)[invar], WEIGHT(*ei));
      }
  else 
    if (invar<0) // invar == ILLEGAL_INT, var is not
      PEITERATE(ei, gen)
        add((*ei)[var], innerVariable->computeValue(*ei), WEIGHT(*ei));
  else // both OK
      PEITERATE(ei, gen)
        add((*ei)[var], (*ei)[invar], WEIGHT(*ei));
}


float TContingencyAttrAttr::p_attr(const TValue &outerValue, const TValue &innerValue) const
{ return p(outerValue)->p(innerValue); }


PDistribution TContingencyAttrAttr::p_attrs(const TValue &outerValue) const
{ return p(outerValue); }


TDomainContingency::TDomainContingency(bool acout)
: classIsOuter(acout)
{}


// Extract TContingency values for all attributes, by iterating through all examples from the generator
TDomainContingency::TDomainContingency(PExampleGenerator gen, const long weightID, bool acout)
: classIsOuter(acout)
{ computeMatrix(gen, weightID); }


// Extract TContingency values for all attributes, by iterating through all examples from the generator
TDomainContingency::TDomainContingency(PExampleGenerator gen, const long weightID, const vector<bool> &attributes, bool acout)
: classIsOuter(acout)
{ computeMatrix(gen, weightID, &attributes); }


void TDomainContingency::computeMatrix(PExampleGenerator gen, const long &weightID, const vector<bool> *attributes, PDomain newDomain)
// IMPORTANT NOTE: When weightID and newDomain are specified, weights are computed from the original examples
// (this is to avoid the need to copy the meta-attributes)
{ PDomain myDomain = newDomain ? newDomain : gen->domain;
  PVariable classVar = myDomain->classVar;

  if (!classVar)
   raiseError("classless domain");

  classes = TDistribution::create(classVar);
  char classType = classVar->varType;

  // Prepare a TContingency for each attribute that has discrete or continuous values

  TValue lastClassValue;
  if (classVar->varType==TValue::INTVAR)
    lastClassValue = TValue(classVar->noOfValues()-1);

  vector<bool>::const_iterator ai, ae;
  if (attributes) {
    ai = attributes->begin();
    ae = attributes->end();
  }
  PITERATE(TVarList, vli, myDomain->attributes) {
    if (attributes) {
      if (ai == ae)
        break;

      if (!*ai++) {
        push_back(NULL);
        continue;
      }
    }
        
    if (classIsOuter)
      push_back(mlnew TContingencyClassAttr(*vli, myDomain->classVar));
    else
      push_back(mlnew TContingencyAttrClass(*vli, myDomain->classVar));

    // if variable and class types are discrete, it initializes distributions to full length
    //   (which will help the copy constructor with estimation below)
    if (((*vli)->varType==TValue::INTVAR) && (classType==TValue::INTVAR) && (lastClassValue.intV >= 0))
      for(int i=0, e=(*vli)->noOfValues(); i!=e; i++)
        back()->add_attrclass(TValue(i), lastClassValue, 0);
  }

  iterator si;
  int Na = myDomain->attributes->size();
  TExample newExample(myDomain);
  TExample::iterator vi, cli;

  PEITERATE(fi, gen) {
    if (newDomain) {
      newDomain->convert(newExample, *fi);
      vi=newExample.begin();
    }
    else
      vi=(*fi).begin();

    cli=vi+Na;
    float xmplWeight=WEIGHT(*fi);
    classes->add(*cli, xmplWeight);
    for(si=begin(); vi!=cli; vi++, si++)
      if (*si)
        (*si)->add_attrclass(*vi, *cli, xmplWeight);
  }
}


/*
TDomainContingency::TDomainContingency(const TDomainContingency &old, PProbabilityEstimator estimator)
: classIsOuter(old.classIsOuter)
{ 
  if (!estimator)
    classes = estimator->operator()(old.classes);
  else {
    classes = CLONE(TDistribution, old.classes);
    classes->normalize();
  }

  if (classIsOuter)
    const_ITERATE(TDomainContingency, di, old)
      push_back(mlnew TContingencyClassAttr(*di, estimator));
  else
    const_ITERATE(TDomainContingency, di, old)
      push_back(mlnew TContingencyAttrClass(*di, estimator));
}
*/


void TDomainContingency::normalize()
{ classes->normalize();
  this_ITERATE(ti) 
    (*ti)->normalize();
}


PDomainDistributions TDomainContingency::getDistributions()
{ PDomainDistributions ddist;
  if (classIsOuter)
    this_ITERATE(ti)
      ddist->push_back((*ti)->innerDistribution);
  else
    this_ITERATE(ti)
      ddist->push_back((*ti)->outerDistribution);
  return ddist;
}


TComputeDomainContingency::TComputeDomainContingency(bool acout)
: classIsOuter(acout)
{}


PDomainContingency TComputeDomainContingency::operator()(PExampleGenerator eg, const long &weightID)
{ return mlnew TDomainContingency(eg, weightID); }

