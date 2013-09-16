#include "stladdon.hpp"
#include "vars.hpp"
#include "distvars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

class TCEvents {
public:
  float failed, censored;
  TCEvents() : failed(0.0), censored(0.0) {}
};

typedef map<float, TCEvents> TTimes;

void survivals(TTimes &times, float &sow, PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID)
{
  const bool outcomemeta = outcomeIndex<0;
  const bool timemeta = timeIndex<0;

  if (!timemeta && (gen->domain->getVar(timeIndex)->varType != TValue::FLOATVAR))
    raiseError("continuous attribute expected for censoring time");
  if (!outcomemeta && (gen->domain->getVar(outcomeIndex)->varType != TValue::INTVAR))
    raiseError("discrete attribute expected for outcome");
  if (failValue.isSpecial() || (failValue.varType!=TValue::INTVAR))
    raiseError("discrete value needs to be specified for the 'failure'");

  const int &failIndex = failValue.intV;

  sow = 0.0;
  PEITERATE(ei, gen) {
    float wei = WEIGHT(*ei);

    TValue &timeval = (*ei)[timeIndex];
    if (timeval.isSpecial())
      continue;
    if (timemeta && timeval.varType != TValue::FLOATVAR)
      raiseError("continuous attribute expected for censoring time");

    TValue &outcomeval = (*ei)[outcomeIndex];
    if (outcomeval.isSpecial())
      continue;
    if (outcomemeta && outcomeval.varType != TValue::INTVAR)
      raiseError("discrete attribute expected for outcome");

    if (outcomeval.intV==failIndex)
      times[timeval.floatV].failed += wei;
    else
      times[timeval.floatV].censored += wei;

    sow += wei;
  }
}


PDistribution kaplanMeier(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID)
{ TTimes times;
  float riskset;
  survivals(times, riskset, gen, outcomeIndex, failValue, timeIndex, weightID);

  TContDistribution *curve = mlnew TContDistribution();
  PDistribution res = curve;

  float survnow = 1.0;
  curve->set(TValue(float(0.0)), survnow);

  ITERATE(TTimes, ti, times)
    if ((*ti).second.failed>0.0) {
      survnow *= (1 - (*ti).second.failed / riskset);
      curve->set(TValue((*ti).first), survnow);
      riskset -= ( (*ti).second.failed + (*ti).second.censored );
    }
    else
      riskset -= (*ti).second.censored;

  return res;
}


PDistribution bayesSurvival(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID, const float &maxTime)
{ TTimes times;
  float N_o_t;
  survivals(times, N_o_t, gen, outcomeIndex, failValue, timeIndex, weightID);

  TTimes::iterator si(times.begin()), se(times.end());
  float failed = 0.0;
  for(; (si!=se) && ((*si).first<=maxTime); si++)
    failed += (*si).second.failed;

  float N_o_inf = failed + (*si).second.censored; // all that have failed + all that have not failed and have been observed for >= maxTime
  if (si!=se)
    while(++si!=se)
      N_o_inf += (*si).second.failed + (*si).second.censored;
  
  if (N_o_inf == 0.0)
    raiseError("bayesSurvival: 'maxTime' too high");

  TContDistribution *curve = mlnew TContDistribution();
  PDistribution res = curve;

  curve->set(TValue(float(0.0)), 1.0);

  float failed_t = 0.0;
  for(si = times.begin(); si!=se; si++) {
    failed_t += (*si).second.failed;

    const float p =  1  -  (failed-failed_t) / (N_o_t-failed_t) * N_o_t / N_o_inf;
    curve->set(TValue((*si).first), p);

    N_o_t -= (*si).second.censored;
  }

  return res;
}
