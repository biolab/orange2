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


#ifndef __RELIEF_HPP
#define __RELIEF_HPP

#include "measures.hpp"

WRAPPER(ExamplesDistance_Relief);
WRAPPER(Domain);


class TNeighbourExample {
public:
  int index;
  float weight;
  float weightEE;

  TNeighbourExample(const int &i, const float &w)
  : index(i), weight(w)
  {}

  TNeighbourExample(const int &i, const float &w, const float &wEE)
  : index(i), weight(w), weightEE(wEE)
  {}
};

class TReferenceExample {
public:
  int index;
  vector<TNeighbourExample> neighbours;
  float nNeighbours;

  TReferenceExample(const int &i = -1)
  : index(i),
    nNeighbours(0.0)
  {}
};


class ORANGE_API TMeasureAttribute_relief : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    float k; //P number of neighbours
    float m; //P number of reference examples

    TMeasureAttribute_relief(int ak=5, int am=100);
    virtual float operator()(PVariable var, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID=0);

    // If attrVals is non-NULL and the values are indeed computed by the thresholdFunction, the caller is 
    // responsible for deallocating the table!
    void thresholdFunction(PVariable var, PExampleGenerator, map<float, float> &res, int weightID = 0, float **attrVals = NULL);

    void thresholdFunction(TFloatFloatList &res, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0);
    float bestThreshold(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0, const float &minSubset = -1);

    PSymMatrix gainMatrix(PVariable var, PExampleGenerator gen, PDistribution, int weightID, int **attrVals, float **attrDistr);
    PIntList bestBinarization(PDistribution &subsets, float &score, PVariable var, PExampleGenerator gen, PDistribution apriorClass = PDistribution(), int weightID = 0, const float &minSubset = -1);

    void reset();

    vector<float> measures;
    int prevExamples, prevWeight;

    // the first int the index of the reference example
    // the inner int-float pairs are indices of neighbours and the corresponding weights
    //   (all indices refer to storedExamples)
    vector<TReferenceExample> neighbourhood;
    PExampleGenerator storedExamples;
    PExamplesDistance distance;
    float ndC, m_ndC;

    void prepareNeighbours(PExampleGenerator, const int &weightID);
    void checkNeighbourhood(PExampleGenerator gen, const int &weightID);
};
    
#endif
