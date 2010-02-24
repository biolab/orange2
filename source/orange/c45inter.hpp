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


#ifndef __C45INTER_HPP
#define __C45INTER_HPP

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;

class ORANGE_API TC45ExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

  PBoolList skip;  //P a boolean list, one element per attribute, denoting which attributes to skip

  TC45ExampleGenerator(const string &datafile, const string &domainFile, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);
  TC45ExampleGenerator(const TC45ExampleGenerator &old);

  virtual bool readExample(TFileExampleIteratorData &, TExample &);

  PDomain readDomain(const string &stem, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);
};

#endif

