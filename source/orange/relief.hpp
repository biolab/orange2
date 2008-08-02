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



class TPairGain {
public:
  float e1, e2, gain;

  TPairGain(const float &ae1, const float &ae2, const float &again)
  : e1(ae1), e2(ae2), gain(again)
  {}
};


class TPairGainAdder : public vector<TPairGain>
{
public:
  void operator()(const float &refVal, const float &neiVal, const float &gain)
  {
    if (refVal < neiVal)
      push_back(TPairGain(refVal, neiVal, gain));
    else
      push_back(TPairGain(neiVal, refVal, gain));
  }
};


float *tabulateContinuousValues(PExampleGenerator gen, const int &weightID, TVariable &variable,
                                float &min, float &max, float &avg, float &N);

class ORANGE_API TMeasureAttribute_relief : public TMeasureAttribute {
public:
    __REGISTER_CLASS

    float k; //P number of neighbours
    float m; //P number of reference examples
    bool checkCachedData; //P tells whether to check the checksum of the data before reusing the cached neighbours

    TMeasureAttribute_relief(int ak=5, int am=100);
    virtual float operator()(PVariable var, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID=0);

    void thresholdFunction(TFloatFloatList &res, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0);
    float bestThreshold(PDistribution &, float &score, PVariable, PExampleGenerator, PDistribution apriorClass=PDistribution(), int weightID = 0, const float &minSubset = -1);

    void pairGains(TPairGainAdder &gains, PVariable var, PExampleGenerator gen, int weightID)
    { thresholdFunction(var, gen, gains, weightID); }

    PSymMatrix gainMatrix(PVariable var, PExampleGenerator gen, PDistribution, int weightID, int **attrVals, float **attrDistr);
    PIntList bestBinarization(PDistribution &subsets, float &score, PVariable var, PExampleGenerator gen, PDistribution apriorClass = PDistribution(), int weightID = 0, const float &minSubset = -1);
    int bestValue(PDistribution &subsetSizes, float &bestScore, PVariable var, PExampleGenerator gen, PDistribution apriorClass, int weightID, const float &minSubset);

    void reset();

    vector<float> measures;
    int prevExamples, prevWeight, prevChecksum, prevK, prevM;

    // the first int the index of the reference example
    // the inner int-float pairs are indices of neighbours and the corresponding weights
    //   (all indices refer to storedExamples)
    vector<TReferenceExample> neighbourhood;
    PExampleGenerator storedExamples;
    PExamplesDistance distance;
    float ndC, m_ndC;

    void prepareNeighbours(PExampleGenerator, const int &weightID);
    void checkNeighbourhood(PExampleGenerator gen, const int &weightID);


    class TFunctionAdder : public map<float, float>
    {
    public:
      inline void addGain(const float &threshold, const float &gain)
      {
        iterator lowerBound = lower_bound(threshold);
        if (lowerBound != end() && (lowerBound->first == threshold))
          lowerBound->second += gain;
        else
          insert(lowerBound, make_pair(threshold, gain));
      }


      void operator()(const float &refVal, const float &neiVal, const float &gain)
      {
        if (refVal < neiVal) {
          addGain(refVal, gain);
          addGain(neiVal, -gain);
        }
        else {
          addGain(neiVal, gain);
          addGain(refVal, -gain);
        }
      }
    };


    // If attrVals is non-NULL and the values are indeed computed by the thresholdFunction, the caller is 
    // responsible for deallocating the table!
    template<class FAdder>
    void thresholdFunction(PVariable var, PExampleGenerator gen, FAdder &adder, int weightID, float **attrVals = NULL)
    {
      if (var->varType != TValue::FLOATVAR)
        raiseError("thresholdFunction can only be computed for continuous attributes");

      checkNeighbourhood(gen, weightID);

      const int attrIdx = gen->domain->getVarNum(var, false);

      const bool regression = gen->domain->classVar->varType == TValue::FLOATVAR;

      if (attrIdx != ILLEGAL_INT) {
        if (attrVals)
          *attrVals = NULL;

        const TExamplesDistance_Relief &rdistance = dynamic_cast<const TExamplesDistance_Relief &>(distance.getReference());
        const TExampleTable &table = dynamic_cast<const TExampleTable &>(gen.getReference());

        adder.clear();
        ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
          const TValue &refVal = table[rei->index][attrIdx];
          if (refVal.isSpecial())
            continue;
          const float &refValF = refVal.floatV;

          ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
            const TValue &neiVal = table[nei->index][attrIdx];
            if (neiVal.isSpecial())
              continue;

            float gain;
            const float attrDist = rdistance(attrIdx, refVal, neiVal);
            if (regression) {
              const float dCdA = nei->weight * attrDist;
              const float dA = nei->weightEE * attrDist;
              gain = dCdA / ndC - (dA - dCdA) / m_ndC;
            }
            else
              gain = nei->weight * attrDist;

            adder(refValF, neiVal.floatV, gain);
          }
        }
      }

      else {
        if (!var->getValueFrom)
          raiseError("attribute is not among the domain attributes and cannot be computed from them");

        float avg, min, max, N;
        float *precals = tabulateContinuousValues(gen, weightID, var.getReference(), min, max, avg, N);
        if (attrVals)
          *attrVals = precals;

        if ((min != max) && (N > 1e-6)) {
          try {
            const float nor = 1.0 / (min-max);

            adder.clear();

            ITERATE(vector<TReferenceExample>, rei, neighbourhood) {
              const float &refValF = precals[rei->index];
              if (refValF == ILLEGAL_FLOAT)
                continue;

              ITERATE(vector<TNeighbourExample>, nei, rei->neighbours) {
                const float &neiValF = precals[nei->index];
                if (neiValF == ILLEGAL_FLOAT)
                  continue;

                float gain;
                const float attrDist = fabs(refValF - neiValF) * nor;
                if (regression) {
                  const float dCdA = nei->weight * attrDist;
                  const float dC = nei->weightEE * attrDist;
                  gain = dCdA / ndC - (dC - dCdA) / m_ndC;
                }
                else
                  gain = nei->weight * attrDist;

                adder(refValF, neiValF, gain);
              }
            }
          }
          catch (...) {
            if (!attrVals)
              delete precals;
            throw;
          }
        }

        if (!attrVals)
          delete precals;
      }
    }

};
    
#endif
