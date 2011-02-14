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


#include <math.h>
#include "crc.h"

#include "stat.hpp"
#include "random.hpp"
#include "values.hpp"
#include "vars.hpp"
#include "stladdon.hpp"
#include "domain.hpp"
#include "examplegen.hpp"
#include "examples.hpp"


#include "distvars.ppp"

DEFINE_TOrangeVector_classDescription(PDistribution, "TDistributionList", true, ORANGE_API)


#define CHECKVALTYPE(valType) \
 if (! (   (valType==TValue::INTVAR) && supportsDiscrete \
        || (valType==TValue::FLOATVAR) && supportsContinuous)) \
   raiseError("invalid value type");

#define NOT_IMPLEMENTED(x) { raiseError("'%s' is not implemented", x); throw 0; /*just to avoid warnings*/ }


TDistribution::TDistribution()
: unknowns(0.0),
  abs(0.0),
  cases(0.0),
  normalized(false),
  supportsDiscrete(false),
  supportsContinuous(false)
{}


TDistribution::TDistribution(PVariable var)
: variable(var),
  unknowns(0.0),
  abs(0.0),
  cases(0.0),
  normalized(false),
  supportsDiscrete(false),
  supportsContinuous(false)
{}


TDistribution *TDistribution::create(PVariable var)
{ if (!var)
    return NULL;
  if (var->varType==TValue::INTVAR)
    return mlnew TDiscDistribution(var);
  if (var->varType==TValue::FLOATVAR)
    return mlnew TContDistribution(var);

  ::raiseErrorWho("Distribution", "unknown value type");
  return NULL; // to make compiler happy
}


TDistribution *TDistribution::fromGenerator(PExampleGenerator gen, const int &position, const int &weightID)
{
  if (position >= gen->domain->variables->size())
    ::raiseErrorWho("Distribution", "index %i out of range", position);

  PVariable var = gen->domain->variables->at(position);

  if (var->varType == TValue::INTVAR)
    return mlnew TDiscDistribution(gen, position, weightID);

  if (var->varType == TValue::FLOATVAR)
    return mlnew TContDistribution(gen, position, weightID);

  ::raiseErrorWho("Distribution", "unknown value type");
  return NULL; // to make compiler happy
}


TDistribution *TDistribution::fromGenerator(PExampleGenerator gen, PVariable var, const int &weightID)
{
  if (var->varType == TValue::INTVAR)
    return mlnew TDiscDistribution(gen, var, weightID);

  if (var->varType == TValue::FLOATVAR)
    return mlnew TContDistribution(gen, var, weightID);

  ::raiseErrorWho("Distribution", "unknown value type");
  return NULL; // to make compiler happy
}



// General

float TDistribution::compatibility(const TSomeValue &) const
NOT_IMPLEMENTED("compatibility")

bool TDistribution::compatible(const TSomeValue &) const
NOT_IMPLEMENTED("compatible")

int TDistribution::compare(const TSomeValue &) const
NOT_IMPLEMENTED("compare")

TDistribution &TDistribution::operator += (const TDistribution &)
NOT_IMPLEMENTED("+=")

TDistribution &TDistribution::operator -= (const TDistribution &)
NOT_IMPLEMENTED("-=")

TDistribution &TDistribution::operator *= (const TDistribution &)
NOT_IMPLEMENTED("*=")

TDistribution &TDistribution::operator *= (const float &)
NOT_IMPLEMENTED("*=")


// Discrete 

const float &TDistribution::atint(const int &)
NOT_IMPLEMENTED("atint(int)")

const float &TDistribution::atint(const int &) const
NOT_IMPLEMENTED("atint(int)")

void TDistribution::addint(const int &, const float &)
NOT_IMPLEMENTED("addint(int, float)")

void TDistribution::setint(const int &, const float &)
NOT_IMPLEMENTED("add(int, float)")

int TDistribution::randomInt()
NOT_IMPLEMENTED("randomInt()")

int TDistribution::randomInt(const long &)
NOT_IMPLEMENTED("randomInt(long)")

int TDistribution::highestProbIntIndex() const
NOT_IMPLEMENTED("highestProbIntIndex()")

int TDistribution::highestProbIntIndex(const long &) const
NOT_IMPLEMENTED("highestProbIntIndex(int)")

int TDistribution::highestProbIntIndex(const TExample &) const
NOT_IMPLEMENTED("highestProbIntIndex(TExample)")

float TDistribution::p(const int &) const
NOT_IMPLEMENTED("p(int)")

int TDistribution::noOfElements() const
NOT_IMPLEMENTED("noOfElements()")

// Continuous

const float &TDistribution::atfloat(const float &)
NOT_IMPLEMENTED("atfloat(float)")

const float &TDistribution::atfloat(const float &) const
NOT_IMPLEMENTED("atfloat(float)")

void TDistribution::addfloat(const float &, const float &)
NOT_IMPLEMENTED("addfloat(float, float)")

void TDistribution::setfloat(const float &, const float &)
NOT_IMPLEMENTED("add(float, float)")

float TDistribution::randomFloat()
NOT_IMPLEMENTED("randomFloat()")

float TDistribution::randomFloat(const long &)
NOT_IMPLEMENTED("randomFloat(long)")

float TDistribution::highestProbFloatIndex() const
NOT_IMPLEMENTED("highestProbFloatIndex()")

float TDistribution::average() const
NOT_IMPLEMENTED("average()")

float TDistribution::dev() const
NOT_IMPLEMENTED("dev()")

float TDistribution::var() const
NOT_IMPLEMENTED("dev()")

float TDistribution::error() const
NOT_IMPLEMENTED("error()")

float TDistribution::percentile(const float &) const
NOT_IMPLEMENTED("percentile(float)")

float TDistribution::p(const float &) const
NOT_IMPLEMENTED("p(float)")




TDistribution &TDistribution::operator +=(PDistribution other)
{ return operator += (other.getReference()); }


TDistribution &TDistribution::operator -=(PDistribution other)
{ return operator -= (other.getReference()); }

TDistribution &TDistribution::operator *=(PDistribution other)
{ return operator *= (other.getReference()); }



float TDistribution::operator -  (const TSomeValue &v) const 
{ return 1-compatibility(v); }


float TDistribution::operator || (const TSomeValue &v) const
{ return 1-compatibility(v); }


const float &TDistribution::operator[](const TValue &val) const 
{ if (val.isSpecial()) {
    if (variable)
      raiseError("undefined value of attribute '%s'", variable->get_name().c_str());
    else
      raiseError("undefined attribute value");
  }
  CHECKVALTYPE(val.varType);
  return (val.varType==TValue::INTVAR) ? atint(int(val)) : atfloat(float(val));
}


const float &TDistribution::operator[](const TValue &val)
{ if (val.isSpecial()) {
    if (variable)
      raiseError("undefined value of attribute '%s'", variable->get_name().c_str());
    else
      raiseError("undefined attribute value");
  }
  CHECKVALTYPE(val.varType);
  return (val.varType==TValue::INTVAR) ? atint(int(val)) : atfloat(float(val));
}


void TDistribution::add(const TValue &val, const float &p)
{ 
  if (!val.svalV || !variable || !variable->distributed) {
    if (val.isSpecial()) {
      unknowns += p;
      if (!val.svalV || !val.svalV.is_derived_from(TDistribution))
        return;
    }
    else {
      CHECKVALTYPE(val.varType);
      if (val.varType==TValue::INTVAR)
        addint(val.intV, p);
      else
        addfloat(val.floatV, p);
      return;
    }
  }

  if (!val.svalV)
    unknowns += p;

  const TDiscDistribution *ddist = val.svalV.AS(TDiscDistribution);
  if (ddist) {
    if (!supportsDiscrete || variable && ddist->variable && (variable!=ddist->variable))
      raiseError("invalid value type");
    int i = 0;
    const_PITERATE(TDiscDistribution, ddi, ddist)
      addint(i++, *ddi*p);
    return;
  }

  const TContDistribution *cdist = val.svalV.AS(TContDistribution);
  if (cdist) {
    if (!supportsContinuous || variable && ddist->variable && (variable!=ddist->variable))
      raiseError("invalid value type");
    const_PITERATE(TContDistribution, cdi, cdist)
      addfloat((*cdi).first, (*cdi).second*p);
    return;
  }

  raiseError("invalid value type");
}


void TDistribution::set(const TValue &val, const float &p)
{ if (!val.isSpecial()) {
    CHECKVALTYPE(val.varType);
    if (val.varType==TValue::INTVAR)
      setint(val.intV, p);
    else
      setfloat(val.floatV, p);
  }
}


TValue TDistribution::highestProbValue() const 
{ if (supportsDiscrete)
    return TValue(highestProbIntIndex());
  else if (supportsContinuous)
    return TValue(highestProbFloatIndex());
  else
    return TValue();
}


TValue TDistribution::highestProbValue(const long &random) const 
{ if (supportsDiscrete)
    return TValue(highestProbIntIndex(random));
  else if (supportsContinuous)
    return TValue(highestProbFloatIndex());
  else
    return TValue();
}


TValue TDistribution::highestProbValue(const TExample &exam) const 
{ if (supportsDiscrete)
    return TValue(highestProbIntIndex(exam));
  else if (supportsContinuous)
    return TValue(highestProbFloatIndex());
  else
    return TValue();
}


TValue TDistribution::randomValue()
{ if (supportsDiscrete)
    return TValue(randomInt());
  else if (supportsContinuous)
    return TValue(randomFloat());
  else 
    return TValue();
}


TValue TDistribution::randomValue(const long &random)
{ if (supportsDiscrete)
    return TValue(randomInt(random));
  else if (supportsContinuous)
    return TValue(randomFloat(random));
  else 
    return TValue();
}


float TDistribution::p(const TValue &val) const
{ if (val.isSpecial()) {
    if (variable)
      raiseError("undefined value of attribute '%s'", variable->get_name().c_str());
    else
      raiseError("undefined attribute value");
  }
  CHECKVALTYPE(val.varType);
  return (val.varType==TValue::INTVAR) ? p(int(val)) : p(float(val));
}




TDiscDistribution::TDiscDistribution() 
{ supportsDiscrete = true; };


TDiscDistribution::TDiscDistribution(PVariable var) 
: TDistribution(var)
{ if (var->varType!=TValue::INTVAR)
     raiseError("attribute '%s' is not discrete", var->get_name().c_str());
  distribution = vector<float>(var->noOfValues(), 0.0);
  supportsDiscrete = true;
}


TDiscDistribution::TDiscDistribution(int values, float value) 
: distribution(vector<float>(values, value))
{ cases = abs = value*values;
  supportsDiscrete = true;
}


TDiscDistribution::TDiscDistribution(const vector<float> &f) 
: distribution(f)
{ abs = 0.0;
  for (const_iterator fi(begin()), fe(end()); fi!=fe; abs += *(fi++));
  cases = abs;
  supportsDiscrete = true;
}


TDiscDistribution::TDiscDistribution(const float *f, const int &len)
: distribution(f, f+len)
{ abs = 0.0;
  for (const_iterator fi(begin()), fe(end()); fi!=fe; abs += *(fi++));
  cases = abs;
  supportsDiscrete = true;
}


TDiscDistribution::TDiscDistribution(PDistribution other) 
: TDistribution(other.getReference())
{ supportsDiscrete = true; }


TDiscDistribution::TDiscDistribution(PDiscDistribution other) 
: TDistribution(other.getReference())
{ supportsDiscrete = true; }


TDiscDistribution::TDiscDistribution(PExampleGenerator gen, const int &position, const int &weightID)
{
  supportsDiscrete = true;

  if (position >= gen->domain->variables->size())
    raiseError("index %i out of range", position);

  variable = gen->domain->variables->at(position);
  if (variable->varType != TValue::INTVAR)
    raiseError("attribute '%s' is not discrete", variable->get_name().c_str());

  distribution = vector<float>(variable->noOfValues(), 0.0);

  PEITERATE(ei, gen)
    add((*ei)[position], WEIGHT(*ei));
}


TDiscDistribution::TDiscDistribution(PExampleGenerator gen, PVariable var, const int &weightID)
: TDistribution(var)
{
  supportsDiscrete = true;

  if (variable->varType != TValue::INTVAR)
    raiseError("attribute '%s' is not discrete", variable->get_name().c_str());

  distribution = vector<float>(var->noOfValues(), 0.0);

  int position = gen->domain->getVarNum(variable, false);
  if (position != ILLEGAL_INT)
    PEITERATE(ei, gen)
      add((*ei)[position], WEIGHT(*ei));
  else
    if (variable->getValueFrom)
      PEITERATE(ei, gen)
        add(variable->computeValue(*ei), WEIGHT(*ei));
    else
      raiseError("attribute '%s' not in domain and cannot be computed", variable->get_name().c_str());
}


const float &TDiscDistribution::atint(const int &v)
{ int ms = v + 1 - size();
  if (ms>0) {
    reserve(v+1);
    while (ms--)
      push_back(0.0);
  }
  return distribution[v]; 
}


const float &TDiscDistribution::atint(const int &v) const
{ if (!size())
    raiseError("empty distribution");
  if ((v < 0) || (v >= int(size()))) 
    raiseError("value %i out of range 0-%i", v, size()-1);
  return at(v); 
}


void TDiscDistribution::addint(const int &v, const float &w)
{ if ((v<0) || (v>1e6))
    raiseError("invalid value");

  int ms = v+1 - size();
  if (ms>0) {
    reserve(v+1);
    while (ms--)
      push_back(0.0);
  }

  float &val = distribution[v];
  val += w;
  abs += w;
  cases += w;
  normalized = false;
}


void TDiscDistribution::setint(const int &v, const float &w)
{ if ((v<0) || (v>1e6))
    raiseError("invalid value");

  int ms = v+1 - size();
  if (ms>0) {
    reserve(v+1);
    while (ms--)
      push_back(0.0);
  }

  float &val=distribution[v];
  abs += w-val;
  cases += w-val;
  val = w;
  normalized = false;
}


TDistribution &TDiscDistribution::adddist(const TDistribution &other, const float &factor)
{
  const TDiscDistribution *mother=dynamic_cast<const TDiscDistribution *>(&other);
  if (!mother)
    raiseError("wrong type of distribution for +=");

  int ms = mother->size() - size();
  if (ms>0) {
    reserve(mother->size());
    while (ms--)
      push_back(0.0);
  }
  
  iterator ti = begin();
  const_iterator oi = mother->begin(), oe = mother->end();
  while(oi!=oe)
    *(ti++) += *(oi++) * factor;
  abs += mother->abs * factor;
  cases += mother->cases;
  unknowns += mother->unknowns;
  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::operator -=(const TDistribution &other)
{
  const TDiscDistribution *mother=dynamic_cast<const TDiscDistribution *>(&other);
  if (!mother)
    raiseError("wrong type of distribution for -=");

  int ms = mother->size() - size();
  if (ms>0) {
    reserve(mother->size());
    while (ms--)
      push_back(0.0);
  }
  
  iterator ti = begin();
  const_iterator oi = mother->begin(), oe = mother->end();
  while(oi!=oe)
    *(ti++) -= *(oi++);
  abs -= mother->abs;
  cases -= mother->cases;
  unknowns -= mother->unknowns;
  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::adddist(PDistribution other, const float &factor)
{ return adddist(other.getReference(), 1.0); }


TDistribution &TDiscDistribution::operator +=(const TDistribution &other)
{ return adddist(other, 1.0); }


TDistribution &TDiscDistribution::operator +=(PDistribution other)
{ return adddist(other.getReference(), 1.0); }


TDistribution &TDiscDistribution::operator -=(PDistribution other)
{ return operator -= (other.getReference()); }



TDistribution &TDiscDistribution::operator *=(const float &weight)
{ for(iterator di(begin()); di!=end(); (*(di++)) *= weight);
  abs *= weight;
  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::operator *=(const TDistribution &other)
{ 
  const TDiscDistribution *mother=dynamic_cast<const TDiscDistribution *>(&other);
  if (!mother)
    raiseError("wrong type of distribution for *=");

  abs = 0.0;
  iterator di = begin(), de = end();
  const_iterator di2 = mother->begin(), de2 = mother->end();
  while ((di!=de) && (di2!=de2))
    abs += (*(di++) *= *(di2++));

  if (di!=de)
    erase(di, de);

  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::operator *= (PDistribution other)
{ return operator *= (other.getReference()); }


TDistribution &TDiscDistribution::operator /=(const TDistribution &other)
{ const TDiscDistribution *mother=dynamic_cast<const TDiscDistribution *>(&other);
  if (!mother)
    raiseError("wrong type of distribution for /=");

  abs = 0.0;
  iterator di = begin(), de = end();
  const_iterator di2 = mother->begin(), de2 = mother->end();
  for (; (di!=de) && (di2!=de2); di++, di2++) {
    if ((-1e-20 < *di2) && (*di2 < 1e-20)) {
      if ((*di<-1e-20) || (*di>1e-20))
        raiseError("division by zero in /=");
    }
    else
      abs += (*di /= *di2);
  }

  if (di!=de)
    erase(di, de);

  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::operator /= (PDistribution other)
{ return operator /= (other.getReference()); }


TDistribution &TDiscDistribution::mul(const TDistribution &other, const float &weight)
{ const TDiscDistribution *mother=dynamic_cast<const TDiscDistribution *>(&other);
  if (!mother)
    raiseError("wrong type of distribution for -=");

  abs = 0.0;
  iterator di = begin(), de = end();
  const_iterator di2 = mother->begin(), de2 = mother->end();
  while ((di!=de) && (di2!=de2))
    abs += (*(di++) *= weight * *(di2++));

  if (di!=de)
    erase(di, de);

  normalized = false;
  return *this;
}


TDistribution &TDiscDistribution::mul(PDistribution other, const float &weight)
{ return mul(other.getReference(), weight); }


/*  Returns normalized scalar products of distributions of 'other' and 'this'.
    The result corresponds to a probability that two random values chosen according
    to the given distributions are same. */
float TDiscDistribution::compatibility(const TSomeValue &ot) const
{ const TDiscDistribution *dv=dynamic_cast<const TDiscDistribution *>(&ot);
  if (dv) {
    float sum=0;
    for(const_iterator i1=begin(), i2=dv->begin();
        (i1!=end());
        sum += *(i1++) * *(i2++))
    return sum/abs/dv->abs;
  }

  const TValue *vv=dynamic_cast<const TValue *>(&ot);
  if (   (vv) 
      || (vv->varType==TValue::INTVAR))
    return (vv->intV>int(size())) ? 0.0 : operator[](vv->intV)/abs;
      
  raiseError("can't compare values of different types");
  return 0.0; // to make compilers happy
}


/*  Declared only since it is abstract in TSomeValue.
    Definition is somewhat artificial: compare does a lexicographical comparison of probabilities. */
int  TDiscDistribution::compare(const TSomeValue &ot) const
{ const TDiscDistribution *dv=dynamic_cast<const TDiscDistribution *>(&ot);
  if (!dv)
    raiseError("can't compare values of different types");

  const_iterator i1=begin(), i2=dv->begin();
  for( ; (i1!=end()) && (*i1==*i2); i1++, i2++);
  if (i1==end())
    return 0;
  else 
    if (*i1<*i2)
      return -1;
  return 1;
}


/*  Declared only since it is abstract in TSomeValue.
    Definitions is somewhat artificial: compatible returns true if compatibility>0 (i.e. if there
    is a non-xero probability that a random values with given distributions are same). */
bool  TDiscDistribution::compatible (const TSomeValue &ot) const
{ return (compatibility(ot)>0); }


void TDiscDistribution::normalize()
{ if (!normalized) {
    if (abs) {
      this_ITERATE(dvi)
        *dvi /= abs;
      abs=1.0;
    }
    else 
      if (size()) {
        float p = 1.0/float(size());
        this_ITERATE(dvi)
          *dvi = p;
        abs = 1.0;
      }
   normalized = true;
  }
}


int TDiscDistribution::highestProbIntIndex() const
{
  if (!size())
    return 0;

  int wins = 1;
  int best = 0;
  float bestP = operator[](0);
  int i, e;

  unsigned long crc;
  INIT_CRC(crc);

  for(i = 1, e = int(size()); --e; i++) {
    const float &P = operator[](i);
    add_CRC(P, crc);

    if (P > bestP) {
      best = i;
      bestP = P;
      wins = 1;
    }
    else if (P==bestP)
      wins++;
  }

  if (wins==1)
    return best;

  FINISH_CRC(crc);
  crc &= 0x7fffffff;

  for(i = 0, wins = 1 + crc % wins; wins; i++)
    if (operator[](i)==bestP)
      wins--;

  return i-1;
}


int TDiscDistribution::highestProbIntIndex(const long &random) const
{
  if (!size())
    return 0;

  int wins = 1;
  int best = 0;
  float bestP = operator[](0);
  int i, e;

  for(i = 1, e = int(size()); --e; i++)
    if (operator[](i) > bestP) {
      best = i;
      bestP = operator[](i);
      wins = 1;
    }
    else if (operator[](i)==bestP)
      wins++;

  if (wins==1)
    return best;

  for(i = 0, wins = 1 + random % wins; wins; i++)
    if (operator[](i)==bestP)
      wins--;

  return i-1;
}


int TDiscDistribution::highestProbIntIndex(const TExample &exam) const
{
  if (!size())
    return 0;

  int wins = 1;
  int best = 0;
  float bestP = operator[](0);
  int i, e;

  for(i = 1, e = int(size()); --e; i++)
    if (operator[](i) > bestP) {
      best = i;
      bestP = operator[](i);
      wins = 1;
    }
    else if (operator[](i)==bestP)
      wins++;

  if (wins==1)
    return best;

  wins = 1 + exam.sumValues() % wins;

  i = 0;    
  while (wins)
    if (operator[](i++)==bestP)
      wins--;

  return i-1;
}


float TDiscDistribution::highestProb() const
{
  float best=-1;
  for(int i=0, isize = size(); i<isize; i++)
    if (operator[](i) > best)
      best=i;
  if (best>=0)
    return operator[](best);
  else
    return size() ? 1.0/size() : 0.0;
}


bool TDiscDistribution::noDeviation() const
{ const_this_ITERATE(dvi)
    if (*dvi)
      return *dvi == abs;
  return size()==1;
}
  

int TDiscDistribution::randomInt(const long &random)
{ 
  float ri = (random & 0x7fffffff) / float(0x7fffffff);
  if (!abs || !size())
    raiseError("cannot return a random element of an empty distribution");
  ri = fmod(ri, abs);
  const_iterator di(begin());
  while (ri > *di)
    ri -= *(di++);
  return int(di-begin());
}


int TDiscDistribution::randomInt()
{ 
  if (!abs || !size())
    raiseError("cannot return a random element of an empty distribution");

  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator;

  float ri = randomGenerator->randfloat(abs);
  const_iterator di(begin());
  while (ri > *di)
    ri -= *(di++);
  return int(di-begin());
}


float TDiscDistribution::p(const int &x) const
{ if (!abs) 
    return size() ? 1.0/size() : 0.0;
  if (x>=size())
    return 0.0;
  return atint(x)/abs; 
}

int TDiscDistribution::noOfElements() const
{ return size(); }


int TDiscDistribution::sumValues() const
{ unsigned long crc;
  INIT_CRC(crc);

  const_this_ITERATE(dvi)
      add_CRC(*dvi, crc);

  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}


TContDistribution::TContDistribution()
: sum(0.0),
  sum2(0.0)
{ supportsContinuous = true; }


TContDistribution::TContDistribution(const map<float, float> &dist)
: distribution(dist), 
  sum(0.0),
  sum2(0.0)
{ abs = 0.0;
  this_ITERATE(di) {
    abs+=(*di).second;
    sum+=(*di).second*(*di).first;
    sum2+=(*di).second*(*di).first*(*di).first;
  }
  cases = abs;
  supportsContinuous = true; 
}


TContDistribution::TContDistribution(PVariable var) 
: TDistribution(var),
  sum(0.0),
  sum2(0.0)
{ if (var->varType!=TValue::FLOATVAR)
     raiseError("attribute '%s' is not continuous", var->get_name().c_str());
  supportsContinuous = true; 
}


TContDistribution::TContDistribution(PExampleGenerator gen, const int &position, const int &weightID)
: sum(0.0),
  sum2(0.0)
{
  supportsContinuous = true;

  if (position >= gen->domain->variables->size())
    raiseError("index %i out of range", position);

  variable = gen->domain->variables->at(position);
  if (variable->varType != TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", variable->get_name().c_str());

  PEITERATE(ei, gen)
    add((*ei)[position], WEIGHT(*ei));
}


TContDistribution::TContDistribution(PExampleGenerator gen, PVariable var, const int &weightID)
: TDistribution(var),
  sum(0.0),
  sum2(0.0)
{
  supportsContinuous = true;

  if (variable->varType != TValue::FLOATVAR)
    raiseError("attribute '%s' is not continuous", variable->get_name().c_str());

  int position = gen->domain->getVarNum(variable, false);
  if (position != ILLEGAL_INT)
    PEITERATE(ei, gen)
      add((*ei)[position], WEIGHT(*ei));
  else
    if (variable->getValueFrom)
      PEITERATE(ei, gen)
        add(variable->computeValue(*ei), WEIGHT(*ei));
    else
      raiseError("attribute '%s' not in domain and cannot be computed", variable->get_name().c_str());
}


const float &TContDistribution::atfloat(const float &v)
{ if (find(v)!=end())
    distribution[v]=0;
  return distribution[v]; 
}


const float &TContDistribution::atfloat(const float &v) const
{ const_iterator vi=find(v);
  if (vi==end())
    raiseError("value %5.3f does not exist", v);
  return (*vi).second;
}


void TContDistribution::addfloat(const float &v, const float &w)
{ 
  iterator vi=find(v);
  if (vi==end())
    distribution[v]=w;
  else
    (*vi).second+=w;

  abs += w;
  cases += w;
  sum += w * v;
  sum2 += w * v*v;
  normalized = false;
}


void TContDistribution::setfloat(const float &v, const float &w)
{ 
  iterator vi=find(v);
  if (vi==end()) {
    distribution[v]=w;
    abs += w;
    cases += w;
    sum += w * v;
    sum += w * v*v;
  }
  else {
    float dif = w - (*vi).second;
    abs += dif;
    cases += w;
    sum += dif * v;
    sum2 += dif * v*v;
    (*vi).second += w;
  }
 
  normalized = false;
}


TDistribution &TContDistribution::operator +=(const TDistribution &other)
{
  const TContDistribution *mother = dynamic_cast<const TContDistribution *>(&other);
  if (!mother)
    raiseError("wrong distribution type for +=");

  const_PITERATE(TContDistribution, oi, mother) 
    addfloat((*oi).first, (*oi).second);

  unknowns += mother->unknowns;

  return *this;
}


TDistribution &TContDistribution::operator -=(const TDistribution &other)
{
  const TContDistribution *mother = dynamic_cast<const TContDistribution *>(&other);
  if (!mother)
    raiseError("wrong distribution type for -=");

  const_PITERATE(TContDistribution, oi, mother) 
    addfloat((*oi).first, -(*oi).second);

  unknowns -= mother->unknowns;

  return *this;
}


TDistribution &TContDistribution::operator +=(PDistribution other)
{ return operator += (other.getReference()); }


TDistribution &TContDistribution::operator -=(PDistribution other)
{ return operator -= (other.getReference()); }



TDistribution &TContDistribution::operator *=(const float &weight)
{ for(iterator i(begin()), e(end()); i!=e; (*(i++)).second*=weight);
  abs *= weight;
  sum *= weight;
  sum2 *= weight;
  normalized = false;
  return *this;
}


float TContDistribution::highestProbFloatIndex() const
{
  // Could use sumValues here, but it's too expensive; this should work for distributions that are distributed enough
  long sum = 0;
  { const_this_ITERATE(i)
      sum += *(long *)(&(*i).first) + *(long *)(&(*i).second);
  }

  TSimpleRandomGenerator rg(sum);

  int wins=0;
  const_iterator best;
  const_this_ITERATE(i)
    if (   (wins==0) && ((wins=1)==1)
        || ((*i).second >  (*best).second) && ((wins=1)==1)
        || ((*i).second == (*best).second) && rg.randbool(++wins))
      best = i;

  if (!wins)
    raiseError("cannot compute the modus of an empty distribution");

  return (*best).first;
}


float TContDistribution::highestProb() const
{
  long sum = 0;
  { const_this_ITERATE(i)
      sum += *(long *)(&(*i).first) + *(long *)(&(*i).second);
   }

  TSimpleRandomGenerator rg(sum);

  int wins=0;
  const_iterator best;
  const_this_ITERATE(i)
    if (   (wins==0) && ((wins=1)==1)
        || ((*i).second >  (*best).second) && ((wins=1)==1)
        || ((*i).second == (*best).second) && rg.randbool(++wins))
      best = i;

  if (wins)
    return (*best).second;
  else
    return size() ? 1.0/size() : 0.0;
}


bool TContDistribution::noDeviation() const
{ return size()==1;
}


float TContDistribution::average() const
{ if (!abs)
    if (variable)
      raiseError("cannot compute average ('%s' has no defined values)", variable->get_name().c_str());
    else
      raiseError("cannot compute average (attribute has no defined values)");

  return sum/abs ; 
}


float TContDistribution::dev() const
{ 
  if (abs<=1e-7)
    if (variable)
      raiseError("cannot compute standard deviation ('%s' has no defined values)", variable->get_name().c_str());
    else
      raiseError("cannot compute standard deviation (attribute has no defined values)");

  const float res = sqrt((sum2-sum*sum/abs)/abs);
  return res > 0 ? res : 0.0;
}
  
float TContDistribution::var() const
{
  if (!abs)
    if (variable)
      raiseError("cannot compute variance ('%s' has no defined values)", variable->get_name().c_str());
    else
      raiseError("cannot compute variance (attribute has no defined values)");

  const float res = (sum2-sum*sum/abs)/abs;
  return res > 0 ? res : 0.0;
}
  
float TContDistribution::error() const
{ if (abs <= 1.0)
    return 0.0;
  const float res = sqrt((sum2-sum*sum/abs)/(abs-1) / abs);
  return res > 0 ? res : 0.0;
}


float TContDistribution::percentile(const float &perc) const
{ if ((perc<0) || (perc>100))
    raiseError("invalid percentile");

  if (!size())
    raiseError("empty distribution");

  if (perc==0.0)
    return (*begin()).first;
  
  if (perc==100.0) {
    const_iterator li(end());
    return (*--li).first;
  }

  float togo = abs*perc/100.0;
  const_iterator ths(begin()), prev, ee(end());

  if (ths == ee)
    raiseError("empty distribution");

  while ((ths != ee) && (togo > 0)) {
    togo -= (*ths).second;
    prev = ths;
    ths++;
  }

  if ((togo < 0) || (ths == ee))
    return (*prev).first;

  // togo==0.0 && ths!=ee
  return ((*prev).first + (*ths).first) / 2.0;
}


void TContDistribution::normalize()
{ if (!normalized) {
    if (abs) {
      this_ITERATE(dvi)
        (*dvi).second /= abs;
      sum /= abs;
      sum2 /= abs;
      abs = 1.0;
    }
    else if (size()) {
      float p = 1.0/float(size());
      sum = 0.0;
      sum2 = 0.0;
      this_ITERATE(dvi) {
        (*dvi).second = p;
        sum += (*dvi).first;
        sum2 += sqr((*dvi).first);
      }
      sum /= abs;
      sum2 /= abs;
      abs = 1.0;
    }

    normalized = true;
  }
}


float TContDistribution::randomFloat()
{
  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator;

  float ri = randomGenerator->randfloat(abs);
  const_iterator di(begin());
  while (ri > (*di).first)
    ri -= (*(di++)).first;
  return (*di).second;
}


float TContDistribution::randomFloat(const long &random)
{ 
  float ri = (random & 0x7fffffff) / float(0x7fffffff);
  const_iterator di(begin());
  while (ri > (*di).first)
    ri -= (*(di++)).first;
  return (*di).second;
}


float TContDistribution::p(const float &x) const
{ const_iterator rb = upper_bound(x);
  if (rb==end())
    return 0.0;
  if ((*rb).first==x)
    return (*rb).second;
  if (rb==begin())
    return 0.0;
  const_iterator lb = rb;
  lb--;

  return (*lb).second + (x - (*lb).first) * ((*rb).second - (*lb).second) / ((*rb).first - (*lb).first);
}


int TContDistribution::sumValues() const
{ unsigned long crc;
  INIT_CRC(crc);

  const_this_ITERATE(dvi) {
    add_CRC((*dvi).first, crc);
    add_CRC((*dvi).second, crc);
  }

  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}


TGaussianDistribution::TGaussianDistribution(const float &amean, const float &asigma, const float &anabs)
: mean(amean),
  sigma(asigma)
{
  abs = anabs;
  normalized = true;
  supportsContinuous = true; 
}


TGaussianDistribution::TGaussianDistribution(PDistribution dist)
: mean(dist->average()),
  sigma(sqrt(dist->dev()))
{
 abs = dist->abs;
 normalized = true; 
 supportsContinuous = true; 
}



float TGaussianDistribution::average() const
{ return mean; }


float TGaussianDistribution::var() const
{ return sigma*sigma; }
  

float TGaussianDistribution::dev() const
{ return sigma; }
  

float TGaussianDistribution::error() const
{ return sigma; }
  

void TGaussianDistribution::normalize()
{ abs = 1.0; }


float TGaussianDistribution::highestProbFloatIndex() const
{ return mean; }


#define pi 3.1415926535897931

float TGaussianDistribution::highestProb() const
{ return abs * 1/(sigma * sqrt(2*pi)); }


float TGaussianDistribution::randomFloat()
{  
  if (!randomGenerator)
    randomGenerator = mlnew TRandomGenerator;

  return (float)gasdev((double)mean, (double)sigma, randomGenerator.getReference());
}


float TGaussianDistribution::randomFloat(const long &random)
{  
  TRandomGenerator rg(random);
  return (float)gasdev((double)mean, (double)sigma, rg);
}


float TGaussianDistribution::p(const float &x) const
{ return abs * exp(-sqr((x-mean)/2/sigma)) / (sigma*sqrt(2*pi)); }


bool TGaussianDistribution::noDeviation() const
{ return sigma==0.0; }


int TGaussianDistribution::sumValues() const
{ unsigned long crc;
  INIT_CRC(crc);
  add_CRC(mean, crc);
  add_CRC(sigma, crc);
  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}


TDomainDistributions::TDomainDistributions()
{}


TDomainDistributions::TDomainDistributions(PExampleGenerator gen, const long weightID, bool skipDiscrete, bool skipContinuous)
{
  reserve(gen->domain->variables->size());
  PITERATE(TVarList, vi, gen->domain->variables) {
    bool compute;
    if ((*vi)->varType == TValue::INTVAR)
      compute = !skipDiscrete;
    else if ((*vi)->varType == TValue::FLOATVAR)
      compute = !skipContinuous;
    else
      compute = false;
    
    push_back(compute ? TDistribution::create(*vi) : PDistribution());
  }

  for(TExampleIterator fi(gen->begin()); fi; ++fi) {
    TExample::iterator ei=(*fi).begin();
    float weight=WEIGHT(*fi);
    for(iterator di=begin(); di!=end(); di++, ei++)
      if (*di)
        (*di)->add(*ei, weight);
  }
}


void TDomainDistributions::normalize()
{ this_ITERATE(di)
    (*di)->normalize(); 
}


PDistribution getClassDistribution(PExampleGenerator gen, const long &weightID)
{ if (!gen)
    raiseErrorWho("getClassDistribution", "no examples");

  if (!gen->domain || !gen->domain->classVar)
    raiseErrorWho("getClassDistribution", "invalid example generator or class-less domain");

  PDistribution classDist = TDistribution::create(gen->domain->classVar);
  TDistribution *uclassdist = const_cast<TDistribution *>(classDist.getUnwrappedPtr());
  PEITERATE(ei, gen)
    uclassdist->add((*ei).getClass(), WEIGHT(*ei));
  return classDist;
}

#undef NOT_IMPLEMENTED
#undef CHECKVALTYPE
