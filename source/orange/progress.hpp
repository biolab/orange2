#ifndef __PROGRESS_HPP
#define __PROGRESS_HPP

#include "root.hpp"

class ORANGE_API TProgressCallback : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual bool operator()(const float &, POrange = POrange()) = 0;

  bool operator()(float *&milestone, POrange = POrange());
  static float *milestones(const int totalSteps, const int nMilestones = 100);
};

WRAPPER(ProgressCallback)

#endif
