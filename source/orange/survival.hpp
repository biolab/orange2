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


#ifndef __SURVIVAL_HPP
#define __SURVIVAL_HPP

#include <map>
#include "classify.hpp"

class ORANGE_API TKaplanMeier : public TOrange {
public:
  __REGISTER_CLASS

  typedef map<float, float> TCurveType;
  TCurveType curve;

  TKaplanMeier(PExampleGenerator gen, const int &outcomeIndex, const int &eventIndex, const int &timeIndex, const int &weightID=0);

  void toFailure();
  void toLog();
  void normalizedCut(const float &maxTime=-1.0);

  float operator()(const float &time);
};


WRAPPER(KaplanMeier);


class ORANGE_API TClassifierForKMWeight : public TClassifier {
public:
  __REGISTER_CLASS

  int whichID; //P Id of meta variable for time
  PVariable outcomeVar; //P outcome variable
  int failIndex; //P index of 'fail' value
  PKaplanMeier kaplanMeier; //P Kaplan-Meier curve

  int lastDomainVersion;
  int lastOPos;

  TClassifierForKMWeight();
  TClassifierForKMWeight(PVariable classvar, PKaplanMeier akm, const int &id, PVariable outcome, const int &failindex);
  TClassifierForKMWeight(const TClassifierForKMWeight &);

  virtual TValue operator ()(const TExample &);
};


class ORANGE_API TClassifierForLinearWeight : public TClassifier {
public:
  __REGISTER_CLASS

  int whichID; //P Id of meta variable for time
  PVariable outcomeVar; //P outcome variable
  int failIndex; //P index of 'fail' value
  float maxTime; //P maximal time

  int lastDomainVersion;
  int lastOPos;

  TClassifierForLinearWeight(PVariable classVar, const float &mT, const int &wid, PVariable ovar, const int &fi);
  TClassifierForLinearWeight(const TClassifierForLinearWeight &old);

  virtual TValue operator ()(const TExample &example);
};

#endif
