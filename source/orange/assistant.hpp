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


#ifndef __ASSISTANT_HPP
#define __ASSISTANT_HPP

#include <string>
#include <iostream>

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;


class TAssistantExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

  TAssistantExampleGenerator(const string &, PDomain);
  TExampleIterator begin();
  virtual bool readExample (TFileExampleIteratorData &, TExample &);
};


class TAssistantDomain : public TDomain {
public:
  __REGISTER_CLASS

  vector<vector<float> *> intervals;
  TAssistantDomain();
  TAssistantDomain(const TAssistantDomain &);
  ~TAssistantDomain();

  static PDomain readDomain(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);

protected:
  static list<TAssistantDomain *> knownDomains;
  static TKnownVariables knownVariables;

  static void removeKnownVariable(TVariable *var);
  static void addKnownDomain(TAssistantDomain *domain);

  bool isSameDomain(const TAssistantDomain *original) const;
};

#endif
