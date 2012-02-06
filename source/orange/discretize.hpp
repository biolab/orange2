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


#ifndef __DISCRETIZE_HPP
#define __DISCRETIZE_HPP

#include <vector>
using namespace std;

#include "values.hpp"
#include "transval.hpp"
#include "domain.hpp"
#include "distvars.hpp"

WRAPPER(BasicAttrStat)

WRAPPER(Discretization)
WRAPPER(EquiDistDiscretizer)
WRAPPER(IntervalDiscretizer)


class ORANGE_API TDiscretization : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID=0)=0;
  void setTransformer(PVariable var, PVariable evar);
};


class ORANGE_API TDiscretizer : public TTransformValue {
public:
  __REGISTER_ABSTRACT_CLASS

  /* If you want to avoid rewrapping, you should write this as static and
     pass the discretizer as PDiscretizer. */
  virtual PVariable constructVar(PVariable, float mindiff = 1.0) = 0;

  virtual void getCutoffs(vector<float> &cutoffs) const = 0;
};

WRAPPER(Discretizer)


class ORANGE_API TDomainDiscretization : public TOrange {
public:
  __REGISTER_CLASS

  PDiscretization discretization; //P discretization

  TDomainDiscretization(PDiscretization = PDiscretization());
  virtual PDomain operator()(PExampleGenerator, const long &weightID=0);

protected:
  PDomain equiDistDomain(PExampleGenerator gen);
  PDomain equiNDomain(PExampleGenerator gen, const long &weightID=0);
  PDomain otherDomain(PExampleGenerator gen, const long &weightID=0);
};


class ORANGE_API TEquiDistDiscretizer : public TDiscretizer {
public:
  __REGISTER_CLASS

  int   numberOfIntervals; //P(+n) number of intervals
  float firstCut; //P the first cut-off point
  float step; //P step (width of interval)

  TEquiDistDiscretizer(const int=-1, const float=-1.0, const float=-1.0);

  virtual void transform(TValue &);
  virtual PVariable constructVar(PVariable, float mindiff = 1.0);

  virtual void getCutoffs(vector<float> &cutoffs) const;
};


class ORANGE_API TThresholdDiscretizer : public TDiscretizer {
public:
  __REGISTER_CLASS

  float threshold; //P threshold

  TThresholdDiscretizer(const float &threshold = 0.0);
  virtual void transform(TValue &);

  virtual PVariable constructVar(PVariable, float mindiff = 1.0);

  virtual void getCutoffs(vector<float> &cutoffs) const;
};


class ORANGE_API TIntervalDiscretizer : public TDiscretizer  {
public:
  __REGISTER_CLASS

  PFloatList points; //P cut-off points

  TIntervalDiscretizer();
  TIntervalDiscretizer(PFloatList apoints);
  TIntervalDiscretizer(const string &boundaries);

  virtual void      transform(TValue &);
  PVariable constructVar(PVariable var, float mindiff = 1.0);

  virtual void getCutoffs(vector<float> &cutoffs) const;
};


class ORANGE_API TBiModalDiscretizer : public TDiscretizer {
public:
  __REGISTER_CLASS

  float low; //P low threshold
  float high; //P high threshold

  TBiModalDiscretizer(const float & = 0.0, const float & = 0.0);
  virtual void transform(TValue &);
  PVariable constructVar(PVariable var, float mindiff = 1.0);

  virtual void getCutoffs(vector<float> &cutoffs) const;
};




class ORANGE_API TEquiDistDiscretization : public TDiscretization {
public:
  __REGISTER_CLASS

  int numberOfIntervals; //P(+n) number of intervals

  TEquiDistDiscretization(const int anumber=4);
  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID=0);
  virtual PVariable operator()(PBasicAttrStat, PVariable) const;
};





class ORANGE_API TFixedDiscretization : public TDiscretization {
public:
  __REGISTER_CLASS

  PFloatList points; //P cut-off points

  TFixedDiscretization(TFloatList &apoints);
  TFixedDiscretization(const string &boundaries);

  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID=0);
};



class ORANGE_API TEquiNDiscretization : public TDiscretization {
public:
  __REGISTER_CLASS

  int numberOfIntervals; //P(+n) number of intervals
  bool recursiveDivision; //P find cut-off points by recursive division (default = true)

  TEquiNDiscretization(int anumber =4);
  virtual PVariable operator()(const TContDistribution &, PVariable var) const;
  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID=0);

  void cutoffsByMidpoints(PIntervalDiscretizer discretizer, const TContDistribution &distr, float &mindiff) const;
  void cutoffsByCounting(PIntervalDiscretizer, const TContDistribution &, float &mindiff) const;
  void cutoffsByDivision(PIntervalDiscretizer, const TContDistribution &, float &mindiff) const;
  void cutoffsByDivision(const int &noInt, TFloatList &points, 
                        map<float, float>::const_iterator fbeg, map<float, float>::const_iterator fend,
                        const float &N, float &mindiff) const;
};



class TSimpleRandomGenerator;

class ORANGE_API TEntropyDiscretization : public TDiscretization {
public:
  __REGISTER_CLASS

  int maxNumberOfIntervals; //P maximal number of intervals; default = 0 (no limits)
  bool forceAttribute; //P minimal number of intervals; default = 0 (no limits)

  TEntropyDiscretization();
  typedef map<float, TDiscDistribution> TS;

  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID = 0);
  virtual PVariable operator()(const TS &, const TDiscDistribution &, PVariable, const long &weightID, TSimpleRandomGenerator &rgen) const;

protected:
  void divide(const TS::const_iterator &, const TS::const_iterator &, const TDiscDistribution &,
              float entropy, int k, vector<pair<float, float> > &, TSimpleRandomGenerator &rgen, float &mindiff) const;
};




class ORANGE_API TBiModalDiscretization : public TDiscretization {
public:
  __REGISTER_CLASS

  bool splitInTwo; //P if true (default), flanks are merged into a single interval

  TBiModalDiscretization(const bool = true);
  virtual PVariable operator()(PExampleGenerator, PVariable, const long &weightID=0);
};


#endif
