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


// to include Python.h before STL defines a template set (doesn't work with VC 6.0)
#include "garbage.hpp" 

#include <set>
#include "stladdon.hpp"

#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "table.hpp"

#include "distance.hpp"

#include "nearest.ppp"


#ifdef _MSC_VER
  #pragma warning (disable : 4512) // assigment operator could not be generated (occurs below, due to references)
#endif


class TNNRec {
public:
  float dist;
  long randoff;
  TExample &example;

  TNNRec(TExample &anexample, const int &roff, const float &adist)
  : dist(adist),
    randoff(roff),
    example(anexample)
  {};

  bool operator <(const TNNRec &other) const
  { return (dist==other.dist) ? (randoff<other.randoff) : (dist<other.dist); }

  bool operator !=(const TNNRec &other) const
  { return (dist!=other.dist) || (randoff!=other.randoff); }
};


TFindNearest::TFindNearest(const int anID, const bool is)
: distanceID(anID),
  includeSame(is)
{}



TFindNearest_BruteForce::TFindNearest_BruteForce(PExampleGenerator gen, const int &wei, PExamplesDistance adist, const int anID, const bool is)
: TFindNearest(anID, is),
  distance(adist),
  examples(gen ? mlnew TExampleTable(gen) : NULL),
  weightID(wei)
{}


PExampleGenerator TFindNearest_BruteForce::operator()(const TExample &e, const float &k, bool needsClass)
{ checkProperty(examples);
  checkProperty(distance);

  
  TExample *nex = e.domain != examples->domain ? new TExample(examples->domain, e) : NULL;
  const TExample &tex = nex ? *nex : e;
  PExampleGenerator res;
  
  try {
    TRandomGenerator rgen(e.sumValues());

    needsClass = needsClass && examples->domain->classVar;

    set<TNNRec> NN;
    PEITERATE(ei, examples) {
      if (!(needsClass && (*ei).getClass().isSpecial())) {
        const float dist = distance->operator()(tex, *ei);
        if (includeSame || (dist>0.0))
          NN.insert(TNNRec(*ei, rgen.randlong(), dist));
      }
    }

    PDomain dom = tex.domain;
    // This creates an ExampleTable with a references to 'examples'
    TExampleTable *ret = mlnew TExampleTable(examples, 1);
    res = ret;

    if (k<=0.0)
      ITERATE(set<TNNRec>, in, NN) {
        TExample &exam = (*in).example;
        if (distanceID)
          exam.setMeta(distanceID, TValue((*in).dist));
        ret->addExample(exam);
      }
    else {
      float needs = k;
      ITERATE(set<TNNRec>, in, NN) {
        TExample &exam = (*in).example;
        if (distanceID)
          exam.setMeta(distanceID, TValue((*in).dist));
        ret->addExample(exam);
        if ((needs -= WEIGHT((*in).example)) <= 0.0)
          break;
      }
    }
  }
  catch (...) {
    if (nex)
      delete nex;
    throw;
  }

  return res;
}


TFindNearestConstructor::TFindNearestConstructor(PExamplesDistanceConstructor edist, const bool is)
: distanceConstructor(edist),
  includeSame(is)
{}


TFindNearestConstructor_BruteForce::TFindNearestConstructor_BruteForce(PExamplesDistanceConstructor edist, const bool is)
: TFindNearestConstructor(edist, is)
{}


PFindNearest TFindNearestConstructor_BruteForce::operator()(PExampleGenerator gen, const int &weightID, const int &distanceID)
{ checkProperty(distanceConstructor);
  return mlnew TFindNearest_BruteForce(gen, weightID, distanceConstructor->call(gen, weightID), distanceID, includeSame);
}

