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

#ifndef __DOMAIN_HPP
#define __DOMAIN_HPP

#include <vector>
#include <list>
#include <set>
#include <string>
using namespace std;

#include "meta.hpp"
#include "vars.hpp"
WRAPPER(Domain);


/*  A mapping from one domain to another. It is used by the convert method of the TDomain object which
    owns the TDomainMapping to convert an example from the domain, described in this object, to the owning domain. */
class ORANGE_API TDomainMapping {
public:
  TDomain *domain;
  /*  Positions of attributes from owning domain in the domain, denoted by 'domain' field. For example,
      if position[2]=5, the value of the second attribute from the owning domain is the value of the fifth
      from the 'domain' domain. Negative values represent id's of `domain`'s meta-attributes.
      If position[i]==ILLEGAL_INT, i-th attribute has no corresponding attribute in
      'domain'; convert passes the entire example to the computeValue method of the i-th variable which will
      either deduce its value from the example or return a DK. */
  vector<int> positions;
  vector<pair<int, int> > metaPositions;

  /* These are metas not to copy: metas that are copied through one of the above two or those that appear in
     definition of the 'domain' */
  #ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
  #endif
  set<int> metasNotToCopy;
  #ifdef _MCS_VER
    #pragma warning(pop)
  #endif

  TDomainMapping(TDomain *);
};


class ORANGE_API TDomain : public TOrange {
public:
  __REGISTER_CLASS

  PVariable classVar; //PR class variable
  PVarList attributes; //PR list of attributes, excluding the class
  PVarList variables; //PR list of attributes, including the class at the end of the list

  TMetaVector metas;

  /*  Domain version is set to ++domainVersion each time a classVar, attributes or variables is changed.
      The value of this field is compared to TDomainMapping's field with the same name. */
  int version; //PR unique version identifier; it's changed each time a domain is changed

  #if _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4251)
  #endif

  list<TDomainMapping> knownDomains;
  list<TDomainMapping>::iterator lastDomain;

  list<TDomain *> knownByDomains;

  typedef void TDestroyNotifier(TDomain *, void *);
  typedef pair<TDestroyNotifier *, void *> TDestroyNotification;
  list<TDestroyNotification> destroyNotifiers;

  #ifdef _MSC_VER
    #pragma warning(pop)
  #endif

  TDomain();
  TDomain(PVariable, const TVarList &attributes);
  TDomain(const TVarList &variables);
  TDomain(const TDomain &);
  ~TDomain();


  void domainChangedDispatcher();
  void domainChangedNoticeHandler(TDomain *);

  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  virtual bool addVariable(PVariable var);
  virtual bool addVariable(PVariable var, int position);
  virtual bool delVariable(PVariable var);

  virtual void removeClass();
  virtual void setClass(PVariable var);
  virtual void changeClass(PVariable var);

  int getVarNum(PVariable, bool throwExc=true) const;
  int getVarNum(const string &, bool throwExc=true) const;

  PVariable getVar(int num, bool throwExc=true);
  PVariable getVar(int num, bool throwExc=true) const;

  PVariable getVar(const string &, bool takeMetas=true, bool throwExc=true);
  PVariable getVar(const string &, bool takeMetas=true, bool throwExc=true) const;

  long getMetaNum(const string &, bool throwExc=true) const;
  long getMetaNum(PVariable, bool throwExc=true) const;

  PVariable getMetaVar(const int &idx, bool throwExc=true);
  PVariable getMetaVar(const int &idx, bool throwExc=true) const;

  PVariable getMetaVar(const string &wname, bool throwExc=true);
  PVariable getMetaVar(const string &wname, bool throwExc=true) const;

  const TMetaDescriptor *getMetaDescriptor(const int &idx, bool throwExc=true) const;
  const TMetaDescriptor *getMetaDescriptor(const string &wname, bool throwExc=true) const;
  const TMetaDescriptor *getMetaDescriptor(const PVariable var, bool throwExc=true) const;

  PVariable operator[](const string &name);
  PVariable operator[](const string &name) const;

  virtual void convert(TExample &dest, const TExample &src, bool filterMetas = false);
  virtual void domainHasChanged();

  void afterSet(const char *name);

  PVariable hasDiscreteAttributes(bool checkClass = true) const;
  PVariable hasContinuousAttributes(bool checkClass = true) const;
  PVariable hasOtherAttributes(bool checkClass = true) const;
  
  void addToCRC(unsigned long &crc) const;
  int sumValues() const;
};


#endif
