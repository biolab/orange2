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


#include "vars.hpp"
#include "subsets.ppp"


TSubsetsGenerator::TSubsetsGenerator()
{}


TSubsetsGenerator::TSubsetsGenerator(PVarList vl)
: varList(vl)
{}


TSubsetsGenerator_iterator::TSubsetsGenerator_iterator(PVarList vl)
: varList(vl)
{}


TSubsetsGenerator_constSize::TSubsetsGenerator_constSize(int aB)
: B(aB)
{}


TSubsetsGenerator_constSize::TSubsetsGenerator_constSize(PVarList vl, int aB)
: TSubsetsGenerator(vl),
  B(aB)
{}


PSubsetsGenerator_iterator  TSubsetsGenerator_constSize::operator()()
{
  return new TSubsetsGenerator_constSize_iterator(varList, B);
}


TSubsetsGenerator_constSize_iterator::TSubsetsGenerator_constSize_iterator(PVarList vl, int aB)
: TSubsetsGenerator_iterator(vl),
  moreToCome(!!varList),
  counter(aB, varList ? varList->size() : 0)
{}


bool TSubsetsGenerator_constSize_iterator::operator()(TVarList &subset)
{ 
  if (!moreToCome)
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
: min(amin),
  max(amax)
{}


TSubsetsGenerator_minMaxSize::TSubsetsGenerator_minMaxSize(PVarList vl, int amin, int amax)
: TSubsetsGenerator(vl),
  min(amin),
  max(amax)
{}



PSubsetsGenerator_iterator TSubsetsGenerator_minMaxSize::operator()()
{
  return new TSubsetsGenerator_minMaxSize_iterator(varList, min, max);
}


TSubsetsGenerator_minMaxSize_iterator::TSubsetsGenerator_minMaxSize_iterator(PVarList vl, int amin, int amax)
: TSubsetsGenerator_iterator(vl),
  B(amin),
  max(amax),
  counter(0, 0)
{
  if ((B<=0) || (max<=0))
    raiseError("invalid subset size limits");

  for(counter = TCounter(B, varList->size());
      !counter.reset() && (B<max);
      counter = TCounter(++B, varList->size()));

  moreToCome =  B <= max;
}

  
bool TSubsetsGenerator_minMaxSize_iterator::operator()(TVarList &subset)
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


TSubsetsGenerator_constant::TSubsetsGenerator_constant(PVarList vl, PVarList cons)
: TSubsetsGenerator(vl),
  constant(cons)
{}


PSubsetsGenerator_iterator TSubsetsGenerator_constant::operator()()
{
  return new TSubsetsGenerator_constant_iterator(varList, constant);
}


TSubsetsGenerator_constant_iterator::TSubsetsGenerator_constant_iterator()
: TSubsetsGenerator_iterator(PVarList()),
  moreToCome(false)
{}


TSubsetsGenerator_constant_iterator::TSubsetsGenerator_constant_iterator(PVarList vl, PVarList cons)
: TSubsetsGenerator_iterator(vl),
  constant(cons)
{
  moreToCome = varList || constant;

  if (moreToCome && varList && constant) {
    PITERATE(TVarList, vi, constant)
      if (!exists(varList.getReference(), *vi)) {
        moreToCome = false;
        break;
      }
  }
}


bool TSubsetsGenerator_constant_iterator::operator()(TVarList &subset)
{ 
  if (!moreToCome) 
    return false;
  
  subset = constant ? constant.getReference() : varList.getReference();
  moreToCome = false;
  return true;
}



TSubsetsGenerator_withRestrictions::TSubsetsGenerator_withRestrictions(PSubsetsGenerator sub)
: subGenerator(sub)
{}


TSubsetsGenerator_withRestrictions::TSubsetsGenerator_withRestrictions(PSubsetsGenerator sub, const TVarList &areq, const TVarList &aforb)
: subGenerator(sub),
  required(mlnew TVarList(areq)),
  forbidden(mlnew TVarList(aforb))
{}


PSubsetsGenerator_iterator TSubsetsGenerator_withRestrictions::operator()()
{
  return new TSubsetsGenerator_withRestrictions_iterator(subGenerator->call(), required, forbidden);
}


TSubsetsGenerator_withRestrictions_iterator::TSubsetsGenerator_withRestrictions_iterator()
: TSubsetsGenerator_iterator(PVarList())
{}


TSubsetsGenerator_withRestrictions_iterator::TSubsetsGenerator_withRestrictions_iterator(PSubsetsGenerator_iterator sub, PVarList areq, PVarList aforb)
: TSubsetsGenerator_iterator(sub ? sub->varList : PVarList()),
  subGenerator_iterator(sub),
  required(areq),
  forbidden(aforb)
{
  if (required && forbidden)
    PITERATE(TVarList, ri, required)
      if (!exists(varList.getReference(), *ri) || exists(forbidden.getReference(), *ri))
        subGenerator_iterator = NULL;
}


bool TSubsetsGenerator_withRestrictions_iterator::call(TVarList &varList)
{
  if (!subGenerator_iterator)
    return false;

  while(subGenerator_iterator->call(varList)) {
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
          break; // BAD: we have found such a subsubset that all its elements can be found in subset
      }
      if (ssi==forbiddenSubSubsets->end())
        return true; // YES! No such subsubset.
    }
    else
      return true;
  }

  return false;
}
