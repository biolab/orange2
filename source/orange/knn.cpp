#include <limits>
#include <list>
#include <math.h>

#include "random.hpp"
#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

#include "distvars.hpp"
#include "distance.hpp"
#include "nearest.hpp"
#include "meta.hpp"

#include "knn.ppp"


TkNNLearner::TkNNLearner(const float &ak, PExamplesDistanceConstructor edc)
: k(ak),
  rankWeight(true),
  distanceConstructor(edc)
{}


PClassifier TkNNLearner::operator()(PExampleGenerator gen, const int &weight)
{ if (!gen->domain->classVar)
    raiseError("class-less domain");

  PFindNearest findNearest = TFindNearestConstructor_BruteForce(distanceConstructor ? distanceConstructor : mlnew TExamplesDistanceConstructor_Euclidean(), true)
                               (gen, weight, getMetaID());

  return mlnew TkNNClassifier(gen->domain, weight, k, findNearest, rankWeight, gen->numberOfExamples());
}


TkNNClassifier::TkNNClassifier(PDomain dom, const int &wei, const float &ak, PFindNearest fdist, const bool &rw, const int &nEx)
: TClassifierFD(dom, true),
  findNearest(fdist),
  k(ak),
  rankWeight(rw),
  weightID(wei),
  nExamples(nEx)
{}


PDistribution TkNNClassifier::classDistribution(const TExample &oexam)
{ checkProperty(findNearest);

  TExample exam(domain, oexam);

  const float tk = k ? k : sqrt(float(nExamples));
  PExampleGenerator neighbours = findNearest->call(exam, tk, true);
  PDistribution classDist = TDistribution::create(classVar);

  if (neighbours->numberOfExamples()==1)
    classDist->add((*neighbours->begin()).getClass());

  else {
    if (rankWeight) {
      const float &sigma2 = tk*tk / -log(0.001);
      int rank2 = 1, rankp=1; // rank2 is rank^2, rankp = rank^2 - (rank-1)^2; and, voila, we don't need rank :)
      PEITERATE(ei, neighbours)
        classDist->add((*ei).getClass(), WEIGHT(*ei) * exp(-(rank2 += (rankp+=2))/sigma2));
    }
    else {
      const int &distanceID = findNearest->distanceID;

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

      if (lastwei == 0.0) {
        PEITERATE(ei, neighbours)
          classDist->add((*ei).getClass());
      }
      else {
        const float &sigma2 = lastwei*lastwei / -log(0.001);
        PEITERATE(ei, neighbours) {
          const float &wei = WEIGHT2(*ei, distanceID);
          classDist->add((*ei).getClass(), WEIGHT(*ei) * exp(-wei*wei/sigma2));
        }
      }
    }
  }

  classDist->normalize();
  return classDist;
}
