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


#include <algorithm>
#include <iomanip>
#include "errors.hpp"
#include "stladdon.hpp"
#include "vars.hpp"
#include "domain.hpp"

#include "examplegen.hpp"
#include "values.hpp"
#include "distvars.hpp"

#include "examples.ppp"

TExample::TExample()
: values(NULL),
  values_end(NULL)
{}


TExample::TExample(PDomain dom)
: domain(dom),
  values(NULL),
  values_end(NULL)
{ if (!dom)
    raiseError("example needs domain");

  const int attrs = domain->variables->size();
  TValue *vi = values = mlnew TValue[attrs];
  values_end = values + attrs;
  PITERATE(TVarList, di, dom->variables)
    *(vi++) = (*di)->DK();

  ITERATE(TMetaVector, mi, dom->metas)
    meta.setValue((*mi).id, (*mi).variable->DC());
}


TExample::TExample(const TExample &orig)
: domain(orig.domain),
  meta(orig.meta)
{ if (domain) {
    const int attrs = domain->variables->size();
    TValue *vi = values = mlnew TValue[attrs];
    values_end = values + attrs;
    for(TValue *origi = orig.values, *thisi = values; thisi != values_end; *(thisi++) = *(origi++));
  }
  else values = values_end = NULL;
}


TExample::TExample(PDomain dom, const TExample &orig)
: domain(dom),
  meta(orig.meta)
{ if (!dom)
    raiseError("example needs domain");

  const int attrs = domain->variables->size();
  values = mlnew TValue[attrs];
  values_end = values + attrs;
  domain->convert(*this, orig);
}


TExample::~TExample()
{ mldelete[] values; }


int TExample::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  
  for(TValue *vi = values, *ve = values_end; vi!=ve; vi++)
    if (vi->svalV)
      PVISIT(vi->svalV);

  const_ITERATE(TMetaValues, mi, meta)
    if ((*mi).second.svalV)
      PVISIT((*mi).second.svalV);

  return 0;
}


int TExample::dropReferences()
{
  for(TValue *vi = values, *ve = values_end; vi!=ve; vi++)
    if (vi->svalV)
      vi->svalV.~PSomeValue();

  const_ITERATE(TMetaValues, mi, meta)
    if ((*mi).second.svalV)
      (*mi).second.svalV.~PSomeValue();

  return 0;
}


TExample &TExample::operator =(const TExample &orig)
{ domain = orig.domain;

  if (domain) {
    const int attrs = domain->variables->size();
    values = mlnew TValue[attrs];
    values_end = values + attrs;
    for(TValue *origi = orig.values, *thisi = values; thisi != values_end; *(thisi++) = *(origi++));
  }
  else
    values = values_end = NULL;

  meta = orig.meta;
  return *this;
}


TValue &TExample::operator[] (PVariable &var)
{ return values[domain->getVarNum(var)]; }


const TValue &TExample::operator[] (PVariable &var) const
{ return values[domain->getVarNum(var)]; }


TValue &TExample::operator[] (const string &name)
{ return values[domain->getVarNum(name)]; }


const TValue &TExample::operator[] (const string &name) const
{ return values[domain->getVarNum(name)]; }


bool TExample::operator < (const TExample &other) const
{ 
  if (domain != other.domain)
    raiseError("examples are from different domains");
  return compare(other)<0;
}


bool TExample::operator == (const TExample &other) const
{ 
  if (domain != other.domain)
    raiseError("examples are from different domains");

  int Na = domain->variables->size();
  if (!Na)
    return true;
  for (const_iterator vi1(begin()), vi2(other.begin()); (*vi1==*vi2) && --Na; vi1++, vi2++);
  return !Na;
}


int TExample::compare(const TExample &other) const
{ if (domain != other.domain)
    raiseError("examples are from different domains");

  const_iterator i1(begin()), i2(other.begin());
  int Na = domain->variables->size();
  if (!Na)
    return true;

  int comp;
  while (0==(comp= (*(i1++)).compare(*(i2++))) && --Na);
  return comp;
}


bool TExample::compatible(const TExample &other) const
{ if (domain != other.domain)
    raiseError("examples are from different domains");
  
  int Na = domain->variables->size();
  if (!Na)
    return true;
  for (const_iterator i1(begin()), i2(other.begin()); (*i1).compatible(*i2) && --Na; i1++, i2++);
  return !Na;
}


int TExample::sumValues() const
{ int sum = 0;
  TValue *vli = values;
  const_PITERATE(TVarList, vi, domain->attributes)
    if ((*vi)->varType == TValue::INTVAR)
      sum += vli->isSpecial() ? (*vi)->noOfValues() : vli->intV;
    else if (((*vi)->varType == TValue::FLOATVAR) && !vli->isSpecial())
      sum += *(int *)(&vli->floatV);
  return abs(sum);
}
