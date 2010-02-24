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


#include <algorithm>
#include <iomanip>
#include "stladdon.hpp"
#include "vars.hpp"
#include "domain.hpp"

#include "examplegen.hpp"
#include "values.hpp"
#include "distvars.hpp"

#include "examples.ppp"
#include "crc.h"

DEFINE_TOrangeVector_classDescription(PExample, "TExampleList", true, ORANGE_API)

long exampleId = 0;

long getExampleId()
{ return ++exampleId; }
 
TExample::TExample()
: values(NULL),
  values_end(NULL),
  name(NULL),
  id(getExampleId())
{}


TExample::TExample(PDomain dom, bool initMetas)
: domain(dom),
  values(NULL),
  values_end(NULL),
  name(NULL),
  id(getExampleId())
{ if (!dom)
    raiseError("example needs domain");

  const int attrs = domain->variables->size();
  TValue *vi = values = mlnew TValue[attrs];
  values_end = values + attrs;
  PITERATE(TVarList, di, dom->variables)
    *(vi++) = (*di)->DK();

  if (initMetas)
    ITERATE(TMetaVector, mi, dom->metas)
      if (!(*mi).optional)
        setMeta((*mi).id, (*mi).variable->DK());
}


TExample::TExample(const TExample &orig, bool copyMetas)
: domain(orig.domain),
  meta(copyMetas ? orig.meta : TMetaValues()),
  name(orig.name ? new string(*orig.name) : NULL),
  id(orig.id)
{ if (domain) {
    const int attrs = domain->variables->size();
    TValue *vi = values = mlnew TValue[attrs];
    values_end = values + attrs;
    for(TValue *origi = orig.values, *thisi = values; thisi != values_end; *(thisi++) = *(origi++));
  }
  else values = values_end = NULL;
}


TExample::TExample(PDomain dom, const TExample &orig, bool copyMetas)
: domain(dom),
  meta(),
  name(NULL),
  id(orig.id)
{ if (!dom)
    raiseError("example needs a domain");

  const int attrs = domain->variables->size();
  values = mlnew TValue[attrs];
  values_end = values + attrs;
  domain->convert(*this, orig, !copyMetas);
}


void TExample::insertVal(TValue &srcval, PVariable var, const long &metaID, vector<bool> &defined)
{
  int position = var ? domain->getVarNum(var, false) : ILLEGAL_INT;

  if (position != ILLEGAL_INT) {
    // Check if this is an ordinary attribute
    if (position >= 0) {
      if (!mergeTwoValues(values[position], srcval, defined[position]))
        raiseError("ambiguous value of attribute '%s'", var->name.c_str());
      else
        defined[position] = true;
    }
    
    else {
      // Is it meta?
      if (hasMeta(position)) {
        if (!mergeTwoValues(meta[metaID], srcval, true))
          raiseError("ambiguous value for meta-attribute '%s'", var->name.c_str());
      }
      else
        setMeta(position, srcval);
    }
  }


  else {
    /* This attribute is not required in the example.
       But if it is meta and there is no other meta-attribute with same id,
       we shall copy it nevertheless */
    if (metaID && !domain->getMetaVar(metaID, false))
      if (hasMeta(metaID)) {
        if (!mergeTwoValues(meta[metaID], srcval, true))
          raiseError("ambiguous value for meta-attribute %i", position);
      }
      else
        setMeta(metaID  , srcval);
  }
}


TExample::TExample(PDomain dom, PExampleList elist)
: domain(dom),
  name(NULL),
  id(elist->size() ? elist->front()->id : getExampleId())
{
  if (!dom)
    raiseError("example needs a domain");

  const int attrs = domain->variables->size();
  vector<bool> defined(attrs, false);

  TValue *vi = values = mlnew TValue[attrs];
  values_end = values + attrs;
  PITERATE(TVarList, di, dom->variables)
    *(vi++) = (*di)->DK();

  PITERATE(TExampleList, eli, elist) {
    TVarList::iterator vli((*eli)->domain->variables->begin());
    TExample::iterator ei((*eli)->begin()), ee((*eli)->end());
    for(; ei!=ee; ei++, vli++) 
      if (!(*ei).isSpecial())
        insertVal(*ei, *vli, 0, defined);

    set<int> metasNotToCopy;
    ITERATE(TMetaVector, mai, (*eli)->domain->metas) {
      metasNotToCopy.insert((*mai).id);
      if ((*eli)->hasMeta((*mai).id))
        insertVal((*eli)->getMeta((*mai).id), (*mai).variable, (*mai).id, defined);
    }

    set<int>::iterator mend(metasNotToCopy.end());
    ITERATE(TMetaValues, mi, (*eli)->meta)
      if (metasNotToCopy.find((*mi).first)==mend)
        insertVal((*mi).second, PVariable(), (*mi).first, defined);
  }

  ITERATE(TMetaVector, mai, domain->metas)
    if (!(*mai).optional && !hasMeta((*mai).id))
      setMeta((*mai).id, (*mai).variable->DK());
}


TExample::~TExample()
{ 
  mldelete[] values; 
  if (name)
    delete name;
}


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

  delete name;
  name = NULL;

  return 0;
}


TExample &TExample::operator =(const TExample &orig)
{
  if (!orig.domain) {
    values = values_end = NULL;
    domain = PDomain();
  }

  else {
    const int attrs = orig.domain->variables->size();

    if (domain != orig.domain) {
      if (domain->variables->size() != attrs) {
        if (values)
          mldelete[] values;
        values = mlnew TValue[attrs];
        values_end = values + attrs;
      }

      domain = orig.domain;
    }

    for(TValue *origi = orig.values, *thisi = values; thisi != values_end; *(thisi++) = *(origi++));
  }

  meta = orig.meta;
  
  if (name) {
    delete name;
    name = NULL;
  }
  if (orig.name)
    name = new string(*orig.name);

  id = orig.id;
  
  return *this;
}


TValue TExample::getValue(PVariable &var) const
{
  // if there is no getValueFrom, throw an exception
  const int position = domain->getVarNum(var, var->getValueFrom);
  return position != ILLEGAL_INT ? operator[](position) : var->computeValue(*this);
}

TValue &TExample::operator[] (PVariable &var)
{ return operator[](domain->getVarNum(var)); }


const TValue &TExample::operator[] (PVariable &var) const
{ return operator[](domain->getVarNum(var)); }


TValue &TExample::operator[] (const string &name)
{ return operator[](domain->getVarNum(name)); }


const TValue &TExample::operator[] (const string &name) const
{ return operator[](domain->getVarNum(name)); }


const TValue &TExample::missingMeta(const int &i) const
{
  const TMetaDescriptor *md = domain->metas[i];
  if (md)
    if (md->optional)
      return md->variable->DK();
    else
      if (md->variable->name.size())
        raiseError("the value of meta attribute '%s' is missing", md->variable->name.c_str());

  // no descriptor or no name
  raiseError("meta value with id %i is missing", i);
  throw 0;
}


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


int TExample::compare(const TExample &other, const bool ignoreClass) const
{ if (domain != other.domain)
    raiseError("examples are from different domains");

  const_iterator i1(begin()), i2(other.begin());
  int Na = domain->variables->size() - (ignoreClass && domain->classVar ? 1 : 0);
  if (!Na)
    return true;

  int comp;
  while (0==(comp= (*(i1++)).compare(*(i2++))) && --Na);
  return comp;
}


bool TExample::compatible(const TExample &other, const bool ignoreClass) const
{ if (domain != other.domain)
    raiseError("examples are from different domains");
  
  int Na = domain->variables->size() - (ignoreClass && domain->classVar ? 1 : 0);
  if (!Na)
    return true;
  for (const_iterator i1(begin()), i2(other.begin()); (*i1).compatible(*i2) && --Na; i1++, i2++);
  return !Na;
}


#include "stringvars.hpp"

void TExample::addToCRC(unsigned long &crc, const bool includeMetas) const
{
  TValue *vli = values;
  const_PITERATE(TVarList, vi, domain->variables) {
    if ((*vi)->varType == TValue::INTVAR)
      add_CRC((const unsigned long)(vli->isSpecial() ? ILLEGAL_INT : vli->intV), crc);
    else if (((*vi)->varType == TValue::FLOATVAR))
      add_CRC(vli->isSpecial() ? ILLEGAL_FLOAT : vli->floatV, crc);
    else if ((*vi)->varType == STRINGVAR) {
      if (vli->isSpecial() || !vli->svalV)
        add_CRC((const unsigned long)ILLEGAL_INT, crc);
      else
        add_CRC(vli->svalV.AS(TStringValue)->value.c_str(), crc);
    }
    vli++;
  }
  
  if (includeMetas) {
    const_ITERATE(TMetaValues, mi, meta) {
      add_CRC((const unsigned long)(mi->first), crc);
      const TValue &val = mi->second;
      if (val.varType == TValue::INTVAR)
        add_CRC((const unsigned long)(val.isSpecial() ? ILLEGAL_INT : val.intV), crc);
      else if (val.varType == TValue::INTVAR)
        add_CRC(val.isSpecial() ? ILLEGAL_FLOAT: val.floatV, crc);
      else if (val.varType == STRINGVAR) {
        if (val.isSpecial()  || !vli->svalV)
          add_CRC((const unsigned long)ILLEGAL_INT, crc);
        else
          add_CRC(val.svalV.AS(TStringValue)->value.c_str(), crc);
      }
    }
  }
}


int TExample::sumValues(const bool includeMetas) const
{ unsigned long crc;
  INIT_CRC(crc);
  addToCRC(crc, includeMetas);
  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}
