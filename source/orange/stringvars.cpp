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


#include <limits>
#include <stdlib.h>

#include "random.hpp"

#include "domain.hpp"
#include "classify.hpp"
#include "stringvars.ppp"


TStringValue::TStringValue(const string &aval)
: value(aval)
{}


TStringValue::TStringValue(const TStringValue &other)
: TSomeValue(other),
  value(other.value)   
{}


void TStringValue::val2str(string &s) const
{ s = value; }


void TStringValue::str2val(const string &s)
{ value = s; }


#define DYNCAST (dynamic_cast<const TStringValue &>(v)).value

int TStringValue::compare(const TSomeValue &v) const
{ return strcmp(value.c_str(), DYNCAST.c_str()); }


bool TStringValue::compatible(const TSomeValue &v) const
{ return operator ==(v) != 0; }


bool TStringValue::operator < (const TSomeValue &v) const
{ return value<DYNCAST; }


bool TStringValue::operator ==(const TSomeValue &v) const
{ return value==DYNCAST; }


bool TStringValue::operator !=(const TSomeValue &v) const
{ return value!=DYNCAST; }


bool TStringValue::operator > (const TSomeValue &v) const
{ return value>DYNCAST; }

#undef DYNCAST


TStringVariable::TStringVariable()
{}


TStringVariable::TStringVariable(const string &aname)
: TVariable(aname)
{ varType=TValue::OTHERVAR; };



bool TStringVariable::firstValue(TValue &) const
{ raiseError("cannot return the first value of a StringVariable attribute");
  return false;
}


bool TStringVariable::nextValue(TValue &) const
{ raiseError("cannot increase the value of a StringVariable attribute");
  return false; 
}


TValue TStringVariable::randomValue(const int &rand)
{ return TValue(mlnew TStringValue("")); }


int TStringVariable::noOfValues() const
{ return -1; }


void TStringVariable::str2val(const string &valname, TValue &valu)
{ valu = TValue(mlnew TStringValue(valname)); }


void TStringVariable::val2str(const TValue &valu, string &vname) const
{ 
  if (!valu.svalV)
    vname = "";

  const TStringValue *sv = dynamic_cast<const TStringValue *>(valu.svalV.getUnwrappedPtr());
  if (!sv)
    raiseErrorWho("val2str", "invalid value type");

  vname = sv->value;
}
