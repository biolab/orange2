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


#ifndef __RETISINTER_HPP
#define __RETISINTER_HPP

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;

class TRetisExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

  TRetisExampleGenerator(const string &datafile, const string &domainfile, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);
  TExampleIterator begin();
  virtual bool readExample(TFileExampleIteratorData &, TExample &);

  PDomain readDomain(const string &stem, PVarList sourceVars, PDomain sourceDomain, bool dontCheckStored, bool dontStore);

  static void destroyNotifier(TDomain *);

protected:
  static list<TDomain *> knownDomains;
};

#endif