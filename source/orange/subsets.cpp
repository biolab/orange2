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


#include "vars.hpp"
#include "subsets.ppp"


bool TSubsetsGenerator::reset()
{ return true; }


/* This is awkward for compatibility reasons:
   vl can, but does not need to be specified.
   If it is not, there must already exist a varList */
bool TSubsetsGenerator::reset(const TVarList &vl)
{ varList = mlnew TVarList(vl); 
  return true;
}


TSubsetsGenerator_constSize::TSubsetsGenerator_constSize(int aB)
: counter(aB, 0),
  B(aB),
  moreToCome(false)
{}


bool TSubsetsGenerator_constSize::reset(const TVarList &vl)
{ moreToCome = TSubsetsGenerator::reset(vl) && reset();
  return moreToCome;
}


bool TSubsetsGenerator_constSize::reset()
{
  if (!varList)
    moreToCome = false;
  else {
    counter = TCounter(B, varList->size());
    moreToCome = counter.reset();
  }

  return moreToCome;
}


bool TSubsetsGenerator_constSize::nextSubset(TVarList &subset)
{ if (!moreToCome)
    return false;

  if (!varList || (counter.limit != varList->size()))
    raiseError("'limit' and/or 'varList' size manipulated during iteration");

  subset.clear();
  ITERATE(TCounter, ci, counter)
    subset.push_back(varList->at(*ci));

  moreToCome = counter.next();
  return true;
}



TSubsetsGenerator_minMaxSize::TSubsetsGenerator_minMaxSize(int amin, int amax)
: counter(0, 0),
  moreToCome(false),
  min(amin),
  max(amax)
{}


bool TSubsetsGenerator_minMaxSize::reset(const TVarList &vl)
{
  moreToCome = (min<=max) && TSubsetsGenerator::reset(vl) && reset();
  return moreToCome;
}

  
bool TSubsetsGenerator_minMaxSize::reset()
{ 
  if (!varList)
    moreToCome = false;
  else {
    if ((min<=0) || (max<=0))
      raiseError("invalid subset size limits");

    for(B = min, counter = TCounter(B, varList->size());
        !counter.reset() && (B<max);
        counter = TCounter(++B, varList->size()));

    moreToCome =  B<=max;
  }

  return moreToCome;
}


bool TSubsetsGenerator_minMaxSize::nextSubset(TVarList &subset)
{
  if (!moreToCome)
    return false;

  if (!varList || (counter.limit != varList->size()))
    raiseError("'limit' and/or 'varList' size manipulated during iteration");

  subset.clear();
  ITERATE(TCounter, ci, counter)
    subset.push_back(varList->at(*ci));

  // moreToCome is true here (otherwise we'd been thrown out before)
  if (!counter.next())
    do {
      if (B==max) {
        moreToCome = false;
        return true;
      }
      counter = TCounter(++B, varList->size());
    } while (!counter.reset());
    
  return true;
}



TSubsetsGenerator_constant::TSubsetsGenerator_constant()
{}

TSubsetsGenerator_constant::TSubsetsGenerator_constant(const TVarList &con)
 : constant(mlnew TVarList(con)) {}


bool TSubsetsGenerator_constant::reset(const TVarList &vl)
{ moreToCome = TSubsetsGenerator::reset(vl) && reset();
  return moreToCome;
}


bool TSubsetsGenerator_constant::reset()
{
  if (!varList)
    moreToCome = false;
  else {
    PITERATE(TVarList, vi, constant)
      if (!exists(varList.getReference(), *vi)) {
        moreToCome = false;
        break;
      }
  }

  return moreToCome;
}


bool TSubsetsGenerator_constant::nextSubset(TVarList &subset)
{ if (!moreToCome) 
    return false;
  
  subset = constant.getReference();
  moreToCome = false;
  return true;
}


TSubsetsGenerator_withRestrictions::TSubsetsGenerator_withRestrictions(PSubsetsGenerator sub)
 : subGenerator(sub)
 {}

TSubsetsGenerator_withRestrictions::TSubsetsGenerator_withRestrictions
 (PSubsetsGenerator sub, const TVarList &areq, const TVarList &aforb)
 : subGenerator(sub), required(mlnew TVarList(areq)), forbidden(mlnew TVarList(aforb))
 {}



bool TSubsetsGenerator_withRestrictions::reset(const TVarList &vl)
{ 
  return TSubsetsGenerator::reset(vl) && subGenerator->reset(vl) && reset();
}


bool TSubsetsGenerator_withRestrictions::reset()
{
  if (!varList)
    return false;

  if (required && forbidden)
    PITERATE(TVarList, ri, required)
      if (!exists(varList.getReference(), *ri) || exists(forbidden.getReference(), *ri))
        return false;

  return true;
}


bool TSubsetsGenerator_withRestrictions::nextSubset(TVarList &varList)
{
  while(subGenerator->nextSubset(varList)) {
    TVarList::iterator ri, re;

    if (required) {
      for(ri=required->begin(), re=required->end();
         (ri!=re) && (find(varList.begin(), varList.end(), *ri)!=varList.end());
         ri++);
      if (ri!=re)
        continue;
    }

    if (forbidden) {
      for(ri=forbidden->begin(), re=forbidden->end();
         (ri!=re) && (find(varList.begin(), varList.end(), *ri)==varList.end());
         ri++);
      if (ri!=re)
        continue;
    }

    if (forbiddenSubSubsets) {
      TVarListList::iterator ssi(forbiddenSubSubsets->begin()), ssie(forbiddenSubSubsets->end());
      for(; ssi!=ssie; ssi++) {
        ri=(*ssi)->begin();
        re=(*ssi)->end();
        for( ; (ri!=re) && (find(varList.begin(), varList.end(), *ri)!=varList.end()); ri++);
        if (ri==re)
          break; // BAD: we have found such a subsubset, that all its elements can be found in subset
      }
      if (ssi==forbiddenSubSubsets->end())
        return true; // YES! No such subsubset.
    }
    else
      return true;
  }

  return false;
}
