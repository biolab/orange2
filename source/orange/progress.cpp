#include "progress.ppp"

bool TProgressCallback::operator()(float *&milestone, POrange o)
{ return operator()(*((++milestone)++), o); }


float *TProgressCallback::milestones(const int totalSteps, const int nMilestones)
{
  float *milestones = new float[2*(totalSteps+1)];
  float *mi = milestones;
  const float step = float(totalSteps)/nMilestones;
  for(int i = 0; i<=nMilestones; i++) {
    *mi++ = floor(0.5+i*step);
    *mi++ = float(i)/100.0;
  }
  *mi++ = -1;
  *mi++ = 1.0;

  return milestones;
}
