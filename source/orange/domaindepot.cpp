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


#include "vars.hpp"
#include "domain.hpp"

#include "domaindepot.hpp"
#include "stringvars.hpp"


PDomain combineDomains(PDomainList sources, TDomainMultiMapping &mapping)
{
  PVariable classVar;
  // I would use reverse iterators, but don't have them
  TDomainList::const_iterator cri(sources->end()), cre(sources->begin());
  while(!(*--cri)->classVar && (cri!=cre));
  classVar = (*cri)->classVar; // might have stopped at the classvar and reached cre which has none...
      
  TVarList variables;

  mapping.clear();
  vector<pair<int, int> > classMapping;

  int domainIdx = 0;
  TDomainList::const_iterator di(sources->begin()), de(sources->end());
  for(; di!=de; di++, domainIdx++) {

    int varIdx = 0;
    TVarList::const_iterator vi((*di)->variables->begin()), ve((*di)->variables->end());
    for(; vi!=ve; vi++, varIdx++) {
      if (*vi == classVar)
        classMapping.push_back(make_pair(domainIdx, varIdx));
      else {
        TDomainMultiMapping::iterator dmmi(mapping.begin());
        TVarList::const_iterator hvi(variables.begin()), hve(variables.end());
        for(; (hvi!=hve) && (*hvi != *vi); hvi++, dmmi++);
        if (hvi==hve) {
          variables.push_back(*vi);
          mapping.push_back(vector<pair<int, int> >());
          mapping.back().push_back(make_pair(domainIdx, varIdx));
        }
        else
          (*dmmi).push_back(make_pair(domainIdx, varIdx));
      }
    }
  }

  if (classVar)
    mapping.push_back(classMapping);

  TDomain *newDomain = mlnew TDomain(classVar, variables);
  PDomain wdomain = newDomain;

  for(domainIdx = 0, di = sources->begin(); di!=de; domainIdx++, di++)
    const_ITERATE(TMetaVector, mi, (*di)->metas) {
      PVariable metavar = newDomain->getMetaVar((*mi).id, false);
      if (!metavar)
        newDomain->metas.push_back(*mi);
      else
        if (metavar != (*mi).variable)
          raiseError("Id %i represents two different attributes ('%s' and '%s')", (*mi).id, metavar->name.c_str(), (*mi).variable->name.c_str());
    }

  return wdomain;
}


void computeMapping(PDomain destination, PDomainList sources, TDomainMultiMapping &mapping)
{
  mapping.clear();
  const_PITERATE(TVarList, vi, destination->variables) {
    mapping.push_back(vector<pair<int, int> >());
    int domainIdx = 0;
    TDomainList::const_iterator si(sources->begin()), se(sources->end());
    for(; si!=se; si++, domainIdx++) {
      int pos = (*si)->getVarNum(*vi, false);
      if (pos != ILLEGAL_INT)
        mapping.back().push_back(make_pair(domainIdx, pos));
    }
  }
}



TDomainDepot::TAttributeDescription::TAttributeDescription(const string &n, const int &vt, bool ord)
: name(n),
  varType(vt),
  ordered(ord)
{}


TDomainDepot::~TDomainDepot()
{
  ITERATE(list<TDomain *>, di, knownDomains) {
    // this could be done by some remove_if, but I don't intend to fight
    //   all various implementations of STL
    list<TDomain::TDestroyNotification>::iterator src((*di)->destroyNotifiers.begin()), end((*di)->destroyNotifiers.end());
    for(; (src!=end) && ((const TDomainDepot *)((*src).second) != this); src++);
    (*di)->destroyNotifiers.erase(src);
  }
}


void TDomainDepot::destroyNotifier(TDomain *domain, void *data)
{ 
  ((TDomainDepot *)(data))->knownDomains.remove(domain);
}

                  
bool TDomainDepot::checkDomain(const TDomain *domain, 
                               const TAttributeDescriptions *attributes, bool hasClass,
                               const TAttributeDescriptions *metas,
                               int *metaIDs)
{
  // check the number of attributes and meta attributes, and the presence of class attribute
  if (    (domain->variables->size() != attributes->size())
       || (bool(domain->classVar) != hasClass)
       || (metas ? (metas->size() != domain->metas.size()) : domain->metas.size() )
     )
    return false;

  // check the names and types of attributes
  TVarList::const_iterator vi(domain->variables->begin());
  const_PITERATE(TAttributeDescriptions, ai, attributes)
    if (    ((*ai).name != (*vi)->name)
         || ((*ai).varType>0) && ((*ai).varType != (*vi)->varType))
      return false;
    else
      vi++;

  // check the meta attributes if they exist
  if (metas)
    const_PITERATE(TAttributeDescriptions, mi, metas) {
      PVariable var = domain->getMetaVar((*mi).name, false);
      if (    !var
           || (((*mi).varType > 0) && ((*mi).varType != var->varType))
         )
        return false;
      if (metaIDs)
        *(metaIDs++) = domain->getMetaNum((*mi).name, false);
    }

  return true;
}


PDomain TDomainDepot::prepareDomain(const TAttributeDescriptions *attributes, bool hasClass,
                                    const TAttributeDescriptions *metas, PVarList knownVars,
                                    const TMetaVector *knownMetas,
                                    const bool dontStore, const bool dontCheckStored,
                                    bool *domainIsNew, int *metaIDs)
{ 
  if (!dontCheckStored)
    ITERATE(list<TDomain *>, kdi, knownDomains)
      if (checkDomain(*kdi, attributes, hasClass, metas, metaIDs)) {
        if (domainIsNew)
          *domainIsNew = false;
        return *kdi;
      }

  TVarList attrList;
  int foo;
  const_PITERATE(TAttributeDescriptions, ai, attributes) {
    PVariable newvar = makeVariable((*ai).name, (*ai).varType, (*ai).values, foo, knownVars, knownMetas, false, false);
    if ((*ai).ordered)
      newvar->ordered = true;
    attrList.push_back(newvar);
  }

  PDomain newDomain;

  PVariable classVar;
  if (hasClass) {
    classVar = attrList.back();
    attrList.erase(attrList.end()-1);
  }
  
  newDomain = mlnew TDomain(classVar, attrList);

  if (metas)
    const_PITERATE(TAttributeDescriptions, mi, metas) {
      int id;
      PVariable var = makeVariable((*mi).name, (*mi).varType, (*mi).values, id, knownVars, knownMetas, false, true);
      if (!id)
        id = getMetaID();
      newDomain->metas.push_back(TMetaDescriptor(id, var));
      if (metaIDs)
        *(metaIDs++) = id;
    }
    
  if (domainIsNew)
    *domainIsNew = true;

  if (!dontStore) {
    newDomain->destroyNotifiers.push_back(TDomain::TDestroyNotification(&TDomainDepot::destroyNotifier, this));
    knownDomains.push_front(newDomain.getUnwrappedPtr());
  }

  return newDomain;
}


PVariable createVariable(const string &name, const int &varType, PStringList values)
{
  switch (varType) {
    case TValue::INTVAR:  return mlnew TEnumVariable(name, values ? values : PStringList(mlnew TStringList()));
    case TValue::FLOATVAR: return mlnew TFloatVariable(name);
    case STRINGVAR: return mlnew TStringVariable(name);
  }

  if (varType==-1)
    ::raiseErrorWho("makeVariable", "unknown type for attribute '%s'", name.c_str());

  return (TVariable *)NULL;
}


PVariable makeVariable(const string &name, unsigned char varType, PStringList values, int &id, PVarList knownVars, const TMetaVector *metas, bool dontCreateNew, bool preferMetas)
{ if (!preferMetas && knownVars)
    const_PITERATE(TVarList, vi, knownVars)
      if (   ((*vi)->name==name)
          && (    (varType==-1)
               || (varType==STRINGVAR) && (*vi).is_derived_from(TStringVariable)
               || ((*vi)->varType==varType))) {
        id = 0;
        return *vi;
      }

  if (metas)
    const_PITERATE(TMetaVector, mi, metas)
      if (   ((*mi).variable->name == name)
          && (    (varType == -1)
               || (varType==STRINGVAR) && (*mi).variable.is_derived_from(TStringVariable)
               || ((*mi).variable->varType==varType))) {
        id = (*mi).id;
        return (*mi).variable;
      }

  if (preferMetas && knownVars)
    const_PITERATE(TVarList, vi, knownVars)
      if (   ((*vi)->name==name)
          && (    (varType==-1)
               || (varType==STRINGVAR) && (*vi).is_derived_from(TStringVariable)
               || ((*vi)->varType==varType))) {
        id = 0;
        return *vi;
      }
  
  id = 0;

  return dontCreateNew ? PVariable() : createVariable(name, varType, values);
}
