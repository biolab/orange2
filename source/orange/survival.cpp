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


#include <math.h>
#include "stladdon.hpp"

#include "vars.hpp"
#include "meta.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"

#include "survival.ppp"


class Tcevents {
public:
  float failed, censored;
  Tcevents()
  : failed(0.0),
    censored(0.0)
  {}
};

typedef map<float, Tcevents> Ttimes;

TKaplanMeier::TKaplanMeier(PExampleGenerator gen, const int &outcomeIndex, const int &eventIndex, const int &timeIndex, const int &weightID)
{
  Ttimes times; // second==true -> censored  (i.e., !event)
  float sow = 0.0;
  PEITERATE(ei, gen) {
    float wei = WEIGHT(*ei);
    times[fabs((*ei).meta[timeIndex].floatV)].*(((*ei)[outcomeIndex].intV==eventIndex) ? &Tcevents::failed : &Tcevents::censored) += wei;
    sow += wei;
  }

  float survnow = 1.0;
  float riskset = sow;
  curve[0.0] = survnow;
  ITERATE(Ttimes, ti, times)
    if ((*ti).second.failed>0.0) {
      curve[(*ti).first] = (survnow*=(1 - (*ti).second.failed / riskset));
      riskset -= ( (*ti).second.failed + (*ti).second.censored );
    }
    else
      riskset -= (*ti).second.censored;
}


void TKaplanMeier::toFailure()
{ ITERATE(TCurveType, kmi, curve)
    (*kmi).second = 1-(*kmi).second;
}

void TKaplanMeier::toLog()
{ ITERATE(TCurveType, kmi, curve)
    (*kmi).second = -log((*kmi).second);
}


float TKaplanMeier::operator()(const float &time)
{ if (time==-1.0)
    return (*curve.rbegin()).second;
  else {
    TCurveType::iterator first1 = curve.upper_bound(time);
    if (first1==curve.end())
      return (*curve.rbegin()).second;
    else if (first1==curve.begin())
      return 1.0;
    else
      return (*(--first1)).second;
  }
}


void TKaplanMeier::normalizedCut(const float &maxTime)
{ float div;
  TCurveType::iterator first1;
  if (maxTime==-1.0) {
    div = (*curve.rbegin()).second;
    first1 = curve.end();
  }
  else {
    first1 = curve.upper_bound(maxTime);
    if (first1==curve.end()) 
      div = (*curve.rbegin()).second;
    else 
      if (first1==curve.begin())
        div=1;
      else {
        div = (*(--first1)).second;
        first1++;
      }
  }

  if (div==0.0)
    return;

  div = 1.0/div;

  for(TCurveType::iterator kmi(curve.begin()); kmi!=first1; (*(kmi++)).second*=div);
  if (first1==curve.end())
    curve[maxTime]=1.0;
  else {
    (*first1).second = 1.0;
    first1++;
    curve.erase(first1, curve.end());
  }
}

    
TClassifierForKMWeight::TClassifierForKMWeight(PVariable classVar, PKaplanMeier km, const int &wid, PVariable ovar, const int &fi)
: TClassifier(classVar),
  whichID(wid),
  outcomeVar(ovar),
  failIndex(fi),
  kaplanMeier(km),
  lastDomainVersion(-1)
{}


TClassifierForKMWeight::TClassifierForKMWeight(const TClassifierForKMWeight &old)
: TClassifier(old),
  whichID(old.whichID),
  outcomeVar(old.outcomeVar),
  failIndex(old.failIndex),
  kaplanMeier(old.kaplanMeier), 
  lastDomainVersion(old.lastDomainVersion),
  lastOPos(old.lastOPos)
{}


TValue TClassifierForKMWeight::operator ()(const TExample &example)
{ if (example.domain->version!=lastDomainVersion) {
    lastOPos = example.domain->getVarNum(outcomeVar);
    lastDomainVersion = example.domain->version;
  }

  if (!example[lastOPos].isSpecial() && example[lastOPos].intV==failIndex)
    return TValue(float(1));

  TValue tme = example.meta[whichID];
  return TValue (tme.isSpecial() ? float(0.0) : kaplanMeier->operator()(tme.floatV));
}
 


TClassifierForLinearWeight::TClassifierForLinearWeight(PVariable classVar, const float &mT, const int &wid, PVariable ovar, const int &fi)
: TClassifier(classVar),
  whichID(wid),
  outcomeVar(ovar),
  failIndex(fi),
  maxTime(mT),
  lastDomainVersion(-1)
{}


TClassifierForLinearWeight::TClassifierForLinearWeight(const TClassifierForLinearWeight &old)
: TClassifier(old),
  whichID(old.whichID),
  outcomeVar(old.outcomeVar),
  failIndex(old.failIndex),
  maxTime(old.maxTime), 
  lastDomainVersion(old.lastDomainVersion),
  lastOPos(old.lastOPos)
{}


TValue TClassifierForLinearWeight::operator ()(const TExample &example)
{ if (example.domain->version!=lastDomainVersion) {
    lastOPos = example.domain->getVarNum(outcomeVar);
    lastDomainVersion = example.domain->version;
  }

  if (!example[lastOPos].isSpecial() && example[lastOPos].intV==failIndex)
    return TValue(float(1));

  TValue tme = example.meta[whichID];
  return TValue (tme.isSpecial() ? float(0.0)
                                 : ((tme.floatV>maxTime) ? float(1.0)
                                                         : float(tme.floatV/maxTime)));
}
