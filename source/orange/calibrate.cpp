/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

// to include Python.h before STL defines a template set (doesn't work with VC 6.0)
#include "garbage.hpp" 

#include "domain.hpp"
#include "classify.hpp"
#include "examplegen.hpp"

#include "calibrate.ppp"

float TThresholdCA::operator()(PClassifier classifier, PExampleGenerator data, const int &weightID, float &optCA, const int &targetValue, TFloatFloatList *CAs)
{
  if (!data->domain->classVar)
    raiseError("classless domain");
  if (data->domain->classVar != classifier->classVar)
    raiseError("classifier's class variables mismatches the given examples'");
  TEnumVariable *classVar = data->domain->classVar.AS(TEnumVariable);
  if (!classVar)
    raiseError("discrete class expected");

  int wtarget;
  if (targetValue >= 0)
    wtarget = targetValue;
  else if (classVar->baseValue >= 0)
    wtarget = classVar->baseValue;
  else if (classVar->values->size() == 2)
    wtarget = 1;
  else
    raiseError("cannot determine target class: none is given, class is not binary and its 'baseValue' is not set");

  typedef map<float, float> tmfpff;
  tmfpff dists;
  float N = 0.0, corr = 0.0;
  PEITERATE(ei, data) 
    if (!(*ei).getClass().isSpecial()) {
      float wei = WEIGHT(*ei);
      N += wei;
      if ((*ei).getClass().intV == wtarget) {
        corr += wei;
        wei = -wei;
      }

      const float prob = classifier->classDistribution(*ei)->atint(wtarget);
      pair<tmfpff::iterator, bool> elm = dists.insert(make_pair(prob, wei));
      if (!elm.second)
        (*elm.first).second += wei;
    }

  optCA = 0;

  if (dists.size() < 2)
    return 0.5;
    
  float optthresh;
  for(tmfpff::const_iterator ni(dists.begin()), ie(dists.end()), ii(ni++); ni != ie; ii = ni++) {
    corr += (*ii).second;
    if ((corr > optCA) || ((corr == optCA) && ((*ii).first < 0.5))) {
      optCA = corr;
      optthresh = ((*ii).first + (*ni).first) / 2.0;
    }
    if (CAs)
      CAs->push_back(make_pair(((*ii).first + (*ni).first) / 2.0, corr/N));
  }

  optCA /= N;
  return optthresh;
} 
