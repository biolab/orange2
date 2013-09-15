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
{ 
  value = s;
}


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
{ varType = STRINGVAR; }


TStringVariable::TStringVariable(const string &aname)
: TVariable(aname)
{ varType = STRINGVAR; };



bool TStringVariable::firstValue(TValue &) const
{ raiseError("cannot return the first value of a StringVariable attribute");
  return false;
}


bool TStringVariable::nextValue(TValue &) const
{ raiseError("cannot increase the value of a StringVariable attribute");
  return false; 
}


TValue TStringVariable::randomValue(const int &rand)
{ return TValue(mlnew TStringValue(""), STRINGVAR); }


int TStringVariable::noOfValues() const
{ return -1; }


void TStringVariable::str2val(const string &valname, TValue &valu)
{ valu = TValue(mlnew TStringValue(valname), STRINGVAR); }


void TStringVariable::val2str(const TValue &valu, string &vname) const
{ 
  if (special2str(valu, vname))
    return;

  if (!valu.svalV) {
    vname = "";
  }

  else {
    const TStringValue *sv = dynamic_cast<const TStringValue *>(valu.svalV.getUnwrappedPtr());
    if (!sv)
      raiseErrorWho("val2str", "invalid value type");

    vname = sv->value;
  }
}



void TStringVariable::val2filestr(const TValue &val, string &str, const TExample &) const
{
  if (!special2str(val, str)) {
    val2str(val, str);
    if (!str.length())
      str = "\"\"";
  }
}


void TStringVariable::filestr2val(const string &valname, TValue &valu, TExample &)
{
  if (!str2special(valname, valu)) {
    if ((valname.length()>=2) && (valname[0] == '"') && (valname[valname.length()-1] == '"')) {
      valu = TValue(mlnew TStringValue(string(valname.begin()+1, valname.end()-1)), STRINGVAR);
      return;
    }

    valu = TValue(mlnew TStringValue(valname), STRINGVAR);
  }
}
