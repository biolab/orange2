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

#include "distance.hpp"
#include "nearest.hpp"
#include "examplegen.hpp"
#include "table.hpp"
#include "lwr.ppp"

TLWRLearner::TLWRLearner()
: k(10.0),
  rankWeight(false)
{}


TLWRLearner::TLWRLearner(PExamplesDistanceConstructor fnc, PLinRegLearner lrl, const float &ak, bool rw)
: distanceConstructor(fnc),
  linRegLearner(lrl),
  k(ak),
  rankWeight(rw)
{}
  

PClassifier TLWRLearner::operator()(PExampleGenerator gen, const int &weightID)
{
  checkProperty(linRegLearner);

  PFindNearest findNearest = TFindNearestConstructor_BruteForce(distanceConstructor ? distanceConstructor : mlnew TExamplesDistanceConstructor_Euclidean(), true)
                               (gen, weightID, getMetaID());

  return mlnew TLWRClassifier(gen->domain, findNearest, linRegLearner, k, rankWeight, weightID);
}


TLWRClassifier::TLWRClassifier()
: k(10.0)
{}


TLWRClassifier::TLWRClassifier(PDomain dom, PFindNearest fn, PLinRegLearner lrl, const float &ak, bool ur, const int &wid)
: TClassifierFD(dom),
  findNearest(fn),
  linRegLearner(lrl),
  k(ak),
  rankWeight(ur),
  weightID(wid)
{}


TValue TLWRClassifier::operator()(const TExample &ex)
{ 
  checkProperty(findNearest);
  checkProperty(linRegLearner);

  TExample exam(domain, ex);

  PExampleGenerator neighbours = findNearest->call(exam, k);

  if (neighbours->numberOfExamples()==1)
    return (*neighbours->begin()).getClass();

  const int &distanceID = findNearest->distanceID;
  if (rankWeight) {
    const float &sigma2 = k*k / -log(0.001);
    int rank2 = 1, rankp=1; // rank2 is rank^2, rankp = rank^2 - (rank-1)^2; and, voila, we don't need rank :)
    PEITERATE(ei, neighbours)
      (*ei).setMeta(distanceID, TValue(float(WEIGHT(*ei) * exp(-(rank2 += (rankp+=2))/sigma2))));
  }
  else {
    TExample *last;
    TExampleTable *neighble = neighbours.AS(TExampleTable);
    if (neighble)
      last = &neighble->back();
    else {
      // This is not really elegant, but there's no other way to get the last example...
      const TExample *last = NULL;
      { PEITERATE(ei, neighbours)
          last = &*ei;
      }
    }

    float lastwei = WEIGHT2(*last, distanceID);
    const float &sigma2 = lastwei*lastwei / -log(0.001);
    PEITERATE(ei, neighbours) {
      const float &wei = WEIGHT2(*ei, distanceID);
      (*ei).setMeta(distanceID, TValue(float(WEIGHT(*ei) * exp(-wei*wei/sigma2))));
    }
  }

  return linRegLearner->call(neighbours, distanceID)->call(exam);
}
