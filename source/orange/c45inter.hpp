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


#ifndef __C45INTER_HPP
#define __C45INTER_HPP

#include <string>

#include "filegen.hpp"
#include "domain.hpp"
//#include "examples.hpp"

using namespace std;


class TC45ExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

  TC45ExampleGenerator(const string &, PDomain);
  virtual bool readExample(TFileExampleIteratorData &, TExample &);
};



/* TC45Domain differs from its ancestor in that it is initialized
    from the C4.5 .names file. It also has an additional field 'skip'
    to mark attributes which are to be skipped when reading examples. */
class TC45Domain : public TDomain {
public:
  __REGISTER_CLASS

  PBoolList skip; //P a boolean list, one element per attribute, denoting which attributes to skip

  TC45Domain();
  TC45Domain(const TC45Domain &);
  ~TC45Domain();

  static PDomain readDomain(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);

protected:
  static list<TC45Domain *> knownDomains;
  static TKnownVariables knownVariables;

  static void removeKnownVariable(TVariable *var);
  static void addKnownDomain(TC45Domain *domain);

  bool isSameDomain(const TC45Domain *original) const;
};

bool   readC45Atom(TFileExampleIteratorData &, vector<string> &);

#endif

