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
#include "examples.hpp"
#include "examplegen.hpp"
#include "stladdon.hpp"

#include "domain.ppp"


DEFINE_TOrangeVector_classDescription(PDomain, "TDomainList", true, ORANGE_API)

// A counter for version field of domains. It is incremented and copied to version field each time a domain is changed.
int domainVersion=0;

TDomainMapping::TDomainMapping(TDomain *dom)
: domain(dom),
  positions() 
{}


TDomain::TDomain()
: classVar((TVariable *)NULL),
  attributes(mlnew TVarList()),
  variables(mlnew TVarList()),
  classVars(mlnew TVarList()),
  version(++domainVersion),
  lastDomain(knownDomains.end()),
  destroyNotifiers()
{}


TDomain::TDomain(const TVarList &vl)
: classVar(vl.size() ? vl.back() : PVariable()),
  attributes(mlnew TVarList(vl)),
  variables(mlnew TVarList(vl)),
  classVars(mlnew TVarList()),
  version(++domainVersion),
  lastDomain(knownDomains.end()),
  destroyNotifiers()
{ if (attributes->size())
    attributes->erase(attributes->end()-1); 
}


TDomain::TDomain(PVariable va, const TVarList &vl)
: classVar(va),
  attributes(mlnew TVarList(vl)),
  variables(mlnew TVarList(vl)),
  classVars(mlnew TVarList()),
  version(++domainVersion),
  lastDomain(knownDomains.end()),
  destroyNotifiers()
{ 
  if (va)
    variables->push_back(va); 
}


TDomain::TDomain(const TDomain &old)
: TOrange(old),
  classVar(old.classVar), 
  attributes(mlnew TVarList(old.attributes.getReference())),
  variables(mlnew TVarList(old.variables.getReference())), 
  classVars(mlnew TVarList(old.classVars.getReference())),
  metas(old.metas),
  version(++domainVersion),
  knownDomains(),  // don't copy, unless you want to mess with the notifiers..
  lastDomain(knownDomains.end()),
  destroyNotifiers()
{}


TDomain::~TDomain()
{ domainChangedDispatcher(); 
  ITERATE(list<TDestroyNotification>, dni, destroyNotifiers)
    (*(*dni).first)(this, (*dni).second);
}


int TDomain::traverse(visitproc visit, void *arg) const
{ TRAVERSE(TOrange::traverse);
  for(TMetaVector::const_iterator mi(metas.begin()), me(metas.end()); mi!=me; mi++)
    PVISIT((*mi).variable);
  return 0;
}


int TDomain::dropReferences()
{ DROPREFERENCES(TOrange::dropReferences);

  metas.clear();
  domainHasChanged();
  return 0;
}


/*  Must be called whenever the domain is changed, i.e. a variable is added, changed are removed.
    This function increases the version number and clears the knownDomains vector. */
void TDomain::domainHasChanged()
{ version = ++domainVersion;
  domainChangedDispatcher();

  knownDomains.clear();
  knownByDomains.clear();
  lastDomain = knownDomains.end();
}


void TDomain::afterSet(const char *name)
{ 
  if (!strcmp(name, "class_var") 
      || !strcmp(name, "classVar")) {
    if (attributes->size()==variables->size())
      variables->push_back(classVar);
    else
      variables->back() = classVar;
    domainHasChanged();
  }

  TOrange::afterSet(name);
}


bool TDomain::addVariable(PVariable var)
{ if (classVar)
    variables->insert(variables->end()-1, var);
  else
    variables->push_back(var);

  attributes->push_back(var);
  domainHasChanged();
  return true;
}



bool TDomain::addVariable(PVariable var, int position)
{ if (position>int(attributes->size()))
    return false;
  variables->insert(variables->begin()+position, var);
  attributes->insert(attributes->begin()+position, var);
  domainHasChanged();
  return true;
}


void TDomain::removeClass()
{ if (classVar) {
    variables->erase(variables->end()-1);
    classVar = (TVariable *)NULL;
    domainHasChanged();
  }
}


void TDomain::setClass(PVariable var)
{ variables->push_back(var);
  classVar = var; 
  domainHasChanged(); 
}


void TDomain::changeClass(PVariable var)
{ removeClass();
  setClass(var); 
  domainHasChanged();
}


bool TDomain::delVariable(PVariable var)
{ 
  TVarList::iterator ai = find(attributes->begin(), attributes->end(), var);
  if (ai==attributes->end())
    return false;

  TVarList::iterator vi = find(variables->begin(), variables->end(), var);
  if (vi==variables->end())
    return false;

  attributes->erase(ai);
  variables->erase(vi);

  domainHasChanged();
  return true;
}


int TDomain::getVarNum(PVariable var, bool throwExc) const
{ int pos = 0;
  TVarList::const_iterator vi, ve;
  for(vi = variables->begin(), ve = variables->end(); vi!=ve; vi++, pos++)
    if (*vi == var)
      return pos;
  for(vi = classVars->begin(), ve = classVars->end(); vi != ve; vi++, pos++)
      if (*vi == var)
            return pos;

  pos = getMetaNum(var, false);
  if ((pos == ILLEGAL_INT) && throwExc)
    raiseError("attribute '%s' not found", var->get_name().c_str());

  return pos;
}


int TDomain::getVarNum(const string &name, bool throwExc) const
{ int pos = 0;
  TVarList::const_iterator vi, ve;
  for(vi = variables->begin(), ve = variables->end(); vi!=ve; vi++, pos++)
    if ((*vi)->get_name()== name)
      return pos;
  for(vi = classVars->begin(), ve = classVars->end(); vi != ve; vi++, pos++)
        if ((*vi)->get_name() == name)
            return pos;

  pos = getMetaNum(name, false);
  if ((pos == ILLEGAL_INT) && throwExc)
    raiseError("attribute '%s' not found", name.c_str());

  return pos;
}


PVariable TDomain::getVar(int num, bool throwExc) const
{ checkProperty(variables);
  
  if (num>=0) {
      if (num < variables->size())
          return variables->at(num);
      if (num - variables->size() < classVars->size())
          return classVars->at(num - variables->size());
      if (throwExc)
        if (!variables->size())
          raiseError("no attributes in domain");
        else
          raiseError("index %i out of range", num);
      else
        return PVariable();
  }
  else {
    const_ITERATE(TMetaVector, mi, metas)
      if ((*mi).id == num)
        return (*mi).variable;

    if (throwExc)
      raiseError("meta attribute with index %i not in domain", num);
    else
      return PVariable();
  }

  return PVariable();
}


PVariable TDomain::getVar(int num, bool throwExc)
{ checkProperty(variables);
  
  if (num>=0) {
      if (num < variables->size())
          return variables->at(num);

      if (num - variables->size() < classVars->size())
          return classVars->at(num - variables->size());

      if (throwExc)
        if (!variables->size())
          raiseError("no attributes in domain");
        else
          raiseError("index %i out of range", num);
      else
        return PVariable();
  }
  else {
    ITERATE(TMetaVector, mi, metas)
      if ((*mi).id == num)
        return (*mi).variable;

    if (throwExc)
      raiseError("meta attribute with index %i not in domain", num);
    else
      return PVariable();
  }

  return PVariable();
}


PVariable TDomain::getVar(const string &name, bool takeMetas, bool throwExc)
{ 
  PITERATE(TVarList, vi, variables) {
    if ((*vi)->get_name()==name)
      return *vi;
  }
  PITERATE(TVarList, vi2, classVars) {
    if ((*vi2)->get_name()==name)
      return *vi2;
  }
  if (takeMetas)
    ITERATE(TMetaVector, mi, metas)
      if ((*mi).variable->get_name()==name)
        return (*mi).variable;

  if (throwExc)
    raiseError("attribute '%s' not found", name.c_str());

  return PVariable();
}


PVariable TDomain::getVar(const string &name, bool takeMetas, bool throwExc) const
{ const_PITERATE(TVarList, vi, variables) {
    if ((*vi)->get_name()==name)
      return *vi;
  }
  const_PITERATE(TVarList, vi2, classVars) {
    if ((*vi2)->get_name()==name)
      return *vi2;
  }

  if (takeMetas)
    const_ITERATE(TMetaVector, mi, metas)
      if ((*mi).variable->get_name()==name)
        return (*mi).variable;

  if (throwExc)
    raiseError("attribute '%s' not found", name.c_str());

  return PVariable();
}


const TMetaDescriptor *TDomain::getMetaDescriptor(const string &wname, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).variable->get_name()==wname)
      return &*mi;

  if (throwExc)
    raiseError("meta attribute '%s' not found", wname.c_str());
  
  return NULL;
}

const TMetaDescriptor *TDomain::getMetaDescriptor(PVariable var, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).variable==var)
      return &*mi;

  if (throwExc)
    raiseError("meta attribute '%s' not found", var->get_name().c_str());
  
  return NULL;
}


const TMetaDescriptor *TDomain::getMetaDescriptor(const int &idx, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).id==idx)
      return &*mi;

  if (throwExc)
    raiseError("meta attribute with index %i not found", idx);
  
  return NULL;
}



long TDomain::getMetaNum(const string &wname, bool throwExc) const
{ const TMetaDescriptor *mi = getMetaDescriptor(wname, throwExc);
  return mi ? mi->id : ILLEGAL_INT;
}


long TDomain::getMetaNum(PVariable var, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).variable==var)
      return (*mi).id;

  if (throwExc)
    raiseError("meta attribute '%s' not found", var->get_name().c_str());

  return ILLEGAL_INT;
}


PVariable TDomain::getMetaVar(const int &idx, bool throwExc)
{ ITERATE(TMetaVector, mi, metas)
    if ((*mi).id==idx)
      return (*mi).variable;

  if (throwExc)
    raiseError("meta attribute with index %i not found", idx);

  return PVariable();
}


PVariable TDomain::getMetaVar(const int &idx, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).id==idx)
      return (*mi).variable;

  if (throwExc)
    raiseError("meta attribute with index %i not found", idx);

  return PVariable();
}


PVariable TDomain::getMetaVar(const string &wname, bool throwExc) const
{ const_ITERATE(TMetaVector, mi, metas)
    if ((*mi).variable->get_name()==wname)
      return (*mi).variable;

  if (throwExc)
    raiseError("meta attribute '%s' not found", wname.c_str());

  return PVariable();
}


PVariable TDomain::getMetaVar(const string &wname, bool throwExc)
{ ITERATE(TMetaVector, mi, metas)
    if ((*mi).variable->get_name()==wname)
      return (*mi).variable;

  if (throwExc)
    raiseError("meta attribute '%s' not found", wname.c_str());

  return PVariable();
}


PVariable TDomain::operator[](const string &name) const
{ return getVar(name, true, true); }


PVariable TDomain::operator[](const string &name)
{ return getVar(name, true, true); }


/*  Converts the example 'src' to 'dest' from this domain. If example is from the same domain,
    values are copied. If domain is different a corresponding TDomainMapping is found (or constructed
    if necessary). Converting is done by setting i-th value of 'dest' to position[i]-th value of
    'src' or by asking variable[i]->computeValue to deduce its value from 'src'. */
void TDomain::convert(TExample &dest, const TExample &src, bool filterMetas)
{
  dest.id = src.id; 
  if (src.domain==this) {
    int Nv = variables->size();
    TExample::iterator de = dest.begin();
    TExample::const_iterator sr = src.begin();
    while(Nv--)
      *(de++) = *(sr++);
    dest.meta = src.meta;
  }

  else {
    if (lastDomain!=knownDomains.end()) // if not, there are no known domains (otherwise lastDomain would point to one)
      if (src.domain!=(*lastDomain).domain)
        for(lastDomain=knownDomains.begin(); lastDomain!=knownDomains.end(); lastDomain++);

    // no domains or no src.domain
    if (lastDomain==knownDomains.end()) { 
      knownDomains.push_back(TDomainMapping(const_cast<TDomain *>(src.domain.getUnwrappedPtr())));
      const_cast<TExample &>(src).domain->knownByDomains.push_back(const_cast<TDomain *>(this));

      lastDomain = knownDomains.end();
      lastDomain--;

      const_PITERATE(TVarList, vi, variables) {
        const int cvi = src.domain->getVarNum(*vi, false);
        (*lastDomain).positions.push_back(cvi);
        if (cvi<0)
          (*lastDomain).metasNotToCopy.insert(cvi);
      }

      ITERATE(TMetaVector, mvi, metas) {
        const int cvi = src.domain->getVarNum((*mvi).variable, false);
        (*lastDomain).metaPositions.push_back(make_pair(int((*mvi).id), cvi));
        if (cvi<0)
          (*lastDomain).metasNotToCopy.insert(cvi);
      }

      const_ITERATE(TMetaVector, mvio, src.domain->metas)
        (*lastDomain).metasNotToCopy.insert((*mvio).id);
    }

    // Now, lastDomain points to an appropriate mapping
    vector<int>::iterator pi((*lastDomain).positions.begin());
    TVarList::iterator vi(variables->begin());

    TExample::iterator deval(dest.begin());
    for(int Nv = dest.domain->variables->size(); Nv--; pi++, vi++)
      *(deval++) = (*pi == ILLEGAL_INT) ? (*vi)->computeValue(src) : src[*pi];

    TMetaVector::iterator mvi(metas.begin());
    for(vector<pair<int, int> >::const_iterator vpii((*lastDomain).metaPositions.begin()), vpie((*lastDomain).metaPositions.end());
        vpii!=vpie; 
        vpii++, mvi++) {
      if (!(*mvi).optional)
        dest.setMeta((*vpii).first, (*vpii).second==ILLEGAL_INT ? (*mvi).variable->computeValue(src) : src[(*vpii).second]);
      else if ((*vpii).second != ILLEGAL_INT) {
        if (src.hasMeta((*vpii).second))
          dest.setMeta((*vpii).first, src[(*vpii).second]);
      }
      else if ((*mvi).variable->getValueFrom) {
          TValue mval = (*mvi).variable->computeValue(src);
          if (!mval.isSpecial())
            dest.setMeta((*vpii).first, mval);
      }
    }

    if (!filterMetas) {
      set<int>::iterator mend = (*lastDomain).metasNotToCopy.end();
      const_ITERATE(TMetaValues, mi, src.meta)
        if ((*lastDomain).metasNotToCopy.find((*mi).first) == mend)
          dest.setMeta((*mi).first, (*mi).second);
    }
  }
}



void TDomain::domainChangedDispatcher()
{ ITERATE(list<TDomainMapping>, di, knownDomains)
    (*di).domain->domainChangedNoticeHandler(this);

  ITERATE(list<TDomain *>, ki, knownByDomains)
    (*ki)->domainChangedNoticeHandler(this);
}


class eqDom {
public:
  TDomain *domain;

  eqDom(TDomain *dom) : domain(dom) {};
  bool operator==(const TDomainMapping &mp) const
  { return mp.domain==domain; }
};


void TDomain::domainChangedNoticeHandler(TDomain *dom)
{ bool rld = (lastDomain==knownDomains.end()) || ((*lastDomain).domain==dom);

  for(list<TDomainMapping>::iterator li(knownDomains.begin()), ln, le(knownDomains.end()); li!=le; )
    if ((*(ln=li++)).domain==dom)
      knownDomains.erase(ln);

  if (rld)
    lastDomain = knownDomains.end();

  knownByDomains.remove(dom);
}



PVariable TDomain::hasOtherAttributes(bool checkClass) const
{
  const_PITERATE(TVarList, vi, checkClass ? variables : attributes)
    if (((*vi)->varType != TValue::FLOATVAR) && ((*vi)->varType != TValue::INTVAR))
      return *vi;
  return PVariable();
}


PVariable TDomain::hasDiscreteAttributes(bool checkClass) const
{
  const_PITERATE(TVarList, vi, checkClass ? variables : attributes)
    if ((*vi)->varType == TValue::INTVAR)
      return *vi;
  return PVariable();
}


PVariable TDomain::hasContinuousAttributes(bool checkClass) const
{
  const_PITERATE(TVarList, vi, checkClass ? variables : attributes)
    if ((*vi)->varType == TValue::FLOATVAR)
      return *vi;
  return PVariable();
}


#include "crc.h"

void TDomain::addToCRC(unsigned long &crc) const
{
  const_PITERATE(TVarList, vi, variables) {
    add_CRC((*vi)->get_name().c_str(), crc);
    add_CRC((const unsigned char)(*vi)->varType, crc);
    if ((*vi)->varType == TValue::INTVAR)
      PITERATE(TStringList, vli, dynamic_cast<TEnumVariable &>(vi->getReference()).values)
        add_CRC(vli->c_str(), crc);
  }
}


int TDomain::sumValues() const
{ unsigned long crc;
  INIT_CRC(crc);
  addToCRC(crc);
  FINISH_CRC(crc);
  return int(crc & 0x7fffffff);
}
