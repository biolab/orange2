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

  if (!timemeta && (gen->domain->variables->at(timeIndex)->varType != TValue::FLOATVAR))
    raiseError("continuous attribute expected for censoring time");
  if (!outcomemeta && (gen->domain->variables->at(outcomeIndex)->varType != TValue::INTVAR))
    raiseError("discrete attribute expected for outcome");
  if (failValue.isSpecial() || (failValue.varType!=TValue::INTVAR))
    raiseError("discrete value needs to be specified for the 'failure'");

  const int &failIndex = failValue.intV;

  PEITERATE(ei, gen) {
    float wei = WEIGHT(*ei);

    TValue &timeval = timemeta ? (*ei).meta[-timeIndex] : (*ei)[timeIndex];
    if (timeval.isSpecial())
      continue;
    if (timemeta && timeval.varType != TValue::FLOATVAR)
      raiseError("continuous attribute expected for censoring time");

    TValue &outcomeval = outcomemeta ? (*ei).meta[-outcomeIndex] : (*ei)[outcomeIndex];
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

/*
PDistribution relativeSurvival(PExampleGenerator gen, const int &outcomeIndex, TValue &failValue, const int &timeIndex, const int &weightID, const float &maxTime)
{ TTimes times;
  float riskset;
  survivals(times, sow, gen, outcomeIndex, failValue, timeIndex, weightID);

  float above = 0.0;
  TTimes::reverse_iterator si(times.rbegin()), se(times.rend());
  for(; (si!=se) && ((*si).first>maxTime); si++);
    above += (*si).second.failed + (*si).second.censored;

  float losers = 0.0;
  for(; si!=se; si++)
    if ((*ti).second.failed


  
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


*/