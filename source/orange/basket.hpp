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

#ifndef __BASKET_HPP
#define __BASKET_HPP

#include <map>
#include <string>

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;

class ORANGE_API TBasketFeeder : public TOrange {
public:
  __REGISTER_CLASS

  bool dontStore; //P disables items storing
  bool dontCheckStored; //P disables items lookup in the global cache
  PDomain domain; //P domain where the meta attributes are stored
  PDomain sourceDomain; //P domain with items that can be reused

  TBasketFeeder(PDomain sourceDomain, bool dontCheckStored, bool dontStore);

  void addItem(TExample &example, const string &atom, const int &lineno);
  static void clearCache();

protected:
  map<string, int> localStore;

  static map<string, TMetaDescriptor> itemCache;
};

WRAPPER(BasketFeeder);


class ORANGE_API TBasketExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS
  PBasketFeeder basketFeeder;

  TBasketExampleGenerator(const string &datafile, PDomain sourceDomain, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);

  virtual bool readExample(TFileExampleIteratorData &, TExample &);
};

#endif