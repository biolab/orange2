#ifndef __CALIBRATE_HPP
#define __CALIBRATE_HPP

#include "root.hpp"
#include "orvector.hpp"

WRAPPER(Classifier)
WRAPPER(ExampleGenerator)

class ORANGE_API TThresholdCA : public TOrange {
public:
  __REGISTER_CLASS

  float operator()(PClassifier, PExampleGenerator, const int &weighID, float &optCA, const int &targetValue = -1, TFloatFloatList *CAs = NULL);
};

#endif
