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


#ifndef __PREPROCESS_HPP
#define __PREPROCESS_HPP

#include <iostream>
#include <vector>
#include "orvector.hpp"

using namespace std;

WRAPPER(Preprocessor);
WRAPPER(Filter);
class TExampleTable;

class TPreprocessor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual void operator()(vector<PExampleGenerator> &generators, long &weightID)=0;
  virtual PExampleGenerator operator()(PExampleGenerator generators, long &weightID)=0;

protected:
  void addFilterAdapter(PFilter filter, vector<PExampleGenerator> &generators);
  PExampleGenerator filterExamples(PFilter filter, PExampleGenerator generator);
  void replaceWithTable(vector<PExampleGenerator> &generators, PExampleGenerator table);
  void storeToTable(vector<PExampleGenerator> &generators, PDomain domain=PDomain());
};


// The below comment prevents pyprops from matching the line
#define DECLARE(x) \
class TPreprocessor_##x : public TPreprocessor \
{ public: \
/**/__REGISTER_CLASS \
    TPreprocessor_##x(const string & ="", const string & =""); \
    virtual void operator()(vector<PExampleGenerator> &, long &weightID); \
    virtual PExampleGenerator operator()(PExampleGenerator, long &weightID); \



#define CONSTRUCTOR(x) TPreprocessor_##x::TPreprocessor_##x(const string &errorStr, const string &parameters)
#define OPERATOR(x)    void TPreprocessor_##x::operator()(vector<PExampleGenerator> &generators, long &weightID)

#define DIRECT_OPERATOR(x)    PExampleGenerator TPreprocessor_##x::operator()(PExampleGenerator generator, long &weightID)

class TProgArguments;


class TPreprocess : public TOrangeVector<PPreprocessor> {
public:
  __REGISTER_CLASS

  TPreprocess();
  TPreprocess(istream &);
  TPreprocess(const TProgArguments &, const string &argname);

  void readStream(istream &istr);
  void addAdapter(const string &line, const string &);
  void addAdapter(PPreprocessor);

  virtual PExampleGenerator operator()(PExampleGenerator, long &weightID, const long &aweightID=0);
};

WRAPPER(Preprocess)

#endif
