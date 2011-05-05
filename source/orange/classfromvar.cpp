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

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "transval.hpp"

#include "classfromvar.ppp"


inline TValue processValue(PTransformValue &transformer, const TValue &val, const PDistribution &distributionForUnknown, bool transformUnknowns)
{
  if (!val.isSpecial() || transformUnknowns)
    return transformer ? transformer->call(val) : val;

  if (distributionForUnknown) {
    PDistribution distr = CLONE(TDistribution, distributionForUnknown);
    distr->normalize();
    return TValue(distr, val.varType, val.valueType);
  }
  
  return val;
}


TClassifierFromVar::TClassifierFromVar(PVariable acv, PDistribution dun)
: TClassifier(acv),
  whichVar(acv),
  transformer(),
  distributionForUnknown(dun),
  transformUnknowns(true),
  lastDomainVersion(-1)
{}


TClassifierFromVar::TClassifierFromVar(PVariable acv, PVariable awhichVar, PDistribution dun)
: TClassifier(acv),
  whichVar(awhichVar),
  transformer(),
  distributionForUnknown(dun),
  transformUnknowns(true),
  lastDomainVersion(-1)
{}


TClassifierFromVar::TClassifierFromVar(PVariable acv, PVariable awhichVar, PDistribution dun, PTransformValue trans)
: TClassifier(acv),
  whichVar(awhichVar),
  transformer(trans),
  distributionForUnknown(dun),
  transformUnknowns(true),
  lastDomainVersion(-1)
{}

TClassifierFromVar::TClassifierFromVar(const TClassifierFromVar &old)
: TClassifier(old),
  whichVar(old.whichVar),
  transformer(old.transformer),
  distributionForUnknown(old.distributionForUnknown),
  transformUnknowns(true),
  lastDomainVersion(-1)
{};


TValue TClassifierFromVar::operator ()(const TExample &example)
{ 
  if ((lastDomainVersion != example.domain->version) || (lastWhichVar != whichVar)) {
    checkProperty(whichVar);

    lastDomainVersion = -1;
    lastWhichVar = whichVar;
    position = 0;

    TVarList::const_iterator vi(example.domain->variables->begin()), ei(example.domain->variables->end());
    for(; (vi!=ei) && (*vi!=whichVar); vi++, position++);
    if (vi==ei)
      position = -1;
  }

  if (position>=0)
    return processValue(transformer, example[position], distributionForUnknown, transformUnknowns);

  TMetaVector::const_iterator mi(example.domain->metas.begin()), me(example.domain->metas.end());
  for( ; (mi!=me) && ((*mi).variable!=whichVar); mi++);
  if (mi!=me)
    return processValue(transformer, example[(*mi).id], distributionForUnknown, transformUnknowns);

  if (whichVar->getValueFrom)
    return processValue(transformer, whichVar->computeValue(example), distributionForUnknown, transformUnknowns);

  int varType;
  if (distributionForUnknown && distributionForUnknown->variable)
    varType = distributionForUnknown->variable->varType;
  else if (classVar)
    varType = classVar->varType;
  else if (!transformer)
    varType = whichVar->varType;
  else if (distributionForUnknown && distributionForUnknown->supportsDiscrete)
    varType = TValue::INTVAR;
  else if (distributionForUnknown && distributionForUnknown->supportsContinuous)
    varType = TValue::FLOATVAR;
  else
    varType = TValue::NONE;
    
  return TValue(CLONE(TDistribution, distributionForUnknown), varType, valueDK);
}



PDistribution TClassifierFromVar::classDistribution(const TExample &exam)
{ 
  if (computesProbabilities) 
    raiseError("invalid setting of 'computesProbabilities'");

  PDistribution dist;
  if (classVar)
    dist = TDistribution::create(classVar);
  else if (whichVar && !transformer)
    dist = TDistribution::create(whichVar);
  else
    checkProperty(classVar); // prints out the usual error  

  dist->add(operator()(exam));
  return dist;
}


void TClassifierFromVar::predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &classDist)
{
  PVariable cvar;
  if (classVar)
    cvar = classVar;
  else if (whichVar && !transformer)
    cvar = whichVar;
  else
    checkProperty(classVar); // prints out the usual error  

  if (computesProbabilities) {
    classDist = classDistribution(ex);
    val = cvar->varType==TValue::FLOATVAR ? TValue(classDist->average()) : classDist->highestProbValue(ex);
  }
  else {
    val = operator()(ex);
    classDist = TDistribution::create(cvar);
    classDist->add(val);
  }
}



TClassifierFromVarFD::TClassifierFromVarFD(PVariable acv, PDomain dom, const int &p, PDistribution dun, PTransformValue atrans)
: TClassifierFD(dom),
  position(p),
  transformer(atrans),
  distributionForUnknown(dun),
  transformUnknowns(true)
{ classVar = acv; }


TClassifierFromVarFD::TClassifierFromVarFD(const TClassifierFromVarFD &old)
: TClassifierFD(old),
  position(old.position),
  transformer(old.transformer),
  distributionForUnknown(old.distributionForUnknown),
  transformUnknowns(true)
{};


TValue TClassifierFromVarFD::operator ()(const TExample &example)
{ 
  if (position == ILLEGAL_INT)
    raiseError("'position' not set");
  
  if (!domain || (example.domain==domain)) {
    if (position >= example.domain->variables->size())
      raiseError("'position' out of range");
    return processValue(transformer, example[position], distributionForUnknown, transformUnknowns);
  }
  else {
    if (position >= domain->variables->size())
      raiseError("'position' out of range");
    PVariable var = domain->getVar(position);
    return processValue(transformer, example.getValue(var), distributionForUnknown, transformUnknowns);
  }
}


PDistribution TClassifierFromVarFD::classDistribution(const TExample &exam)
{ 
  if (computesProbabilities) 
    raiseError("invalid setting of 'computesProbabilities'");

  TValue val = operator()(exam); // call this first to check the 'position'

  PDistribution dist;
  if (classVar)
    dist = TDistribution::create(classVar);
  else if (!transformer) {
    if (!domain || (exam.domain == domain))
      dist = TDistribution::create(exam.domain->getVar(position));
    else
      dist = TDistribution::create(domain->getVar(position));
  }
  else
    checkProperty(classVar); // prints out the usual error  

  dist->add(val);
  return dist;
}


void TClassifierFromVarFD::predictionAndDistribution(const TExample &ex, TValue &val, PDistribution &classDist)
{
  if (computesProbabilities) {
    classDist = classDistribution(ex);
    val = classDist->supportsContinuous ? TValue(classDist->average()) : classDist->highestProbValue(ex);
  }
  else {
    val = operator()(ex);
    if (!domain || (ex.domain == domain))
      classDist = TDistribution::create(classVar ? classVar : ex.domain->getVar(position));
    else
      classDist = TDistribution::create(classVar ? classVar : domain->getVar(position));
    classDist->add(val);
  }
}
