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
#include "domain.hpp"


#include "examples.hpp"
#include "examplegen.hpp"

#include <vector>
#include <functional>

#include "basstat.ppp"

DEFINE_TOrangeVector_classDescription(PBasicAttrStat, "TBasicAttrStatList")

// Initializes min to max_float, max to min_float, avg, dev and n to 0.
TBasicAttrStat::TBasicAttrStat(PVariable var, const bool &ahold)
: sum(0.0),
  sum2(0.0),
  n(0.0),
  min(numeric_limits<float>::max()),
  max(-numeric_limits<float>::max()),
  avg(0.0),
  dev(0.0),
  variable(var),
  holdRecomputation(ahold)
{}


// Adds an example with value f and weight p; n is increased by p, avg by p*f and dev by p*sqr(f)
void TBasicAttrStat::add(float f, float p)
{ sum += p*f;
  sum2 += p*f*f;
  n += p;
  if (!holdRecomputation && (n>0)) {
    avg = sum/n;
    dev = sqrt(sum2/n - avg*avg);
  }

  if (f<min) 
    min = f;
  if (f>max)
    max = f;
}

void TBasicAttrStat::recompute()
{ if (n>0) {
    avg = sum/n;
    dev = sqrt(sum2/n - avg*avg);
  }
  else
    avg = dev = -1;
}



TDomainBasicAttrStat::TDomainBasicAttrStat()
{}


TDomainBasicAttrStat::TDomainBasicAttrStat(PExampleGenerator gen, const long &weightID)
{ PITERATE(TVarList, vi, gen->domain->variables)
    push_back(((*vi)->varType==TValue::FLOATVAR) ? PBasicAttrStat(mlnew TBasicAttrStat(*vi, true)) : PBasicAttrStat());

  PEITERATE(fi, gen) {
    TExample::iterator ei = (*fi).begin();
    float wei = WEIGHT(*fi);
    for(iterator di(begin()); di!=end(); di++, ei++)
      if (*di && !ei->isSpecial())
        (*di)->add(*ei, wei);
  }

  this_ITERATE(di)
    if (*di) {
      (*di)->holdRecomputation = false;
      (*di)->recompute();
    }
}


/* Removes empty BasicAttrStat (i.e. those at places corresponding to non-continuous attributes */
void TDomainBasicAttrStat::purge()
{ erase(remove_if(begin(), end(), logical_not<PBasicAttrStat>()), end()); }
