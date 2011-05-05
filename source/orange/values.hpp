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


#ifndef __VALUES_HPP
#define __VALUES_HPP

#include <limits>
#include "root.hpp"

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

#undef max
template<class T>int sign(const T &v) { return (v==0) ? 0 : ( (v>0) ? 1: -1); }

/* A more general value holder than TValue. Derived objects will include additional field(s) for
   storing and handling the data, and define abstract methods get a functional class. */
class ORANGE_API TSomeValue : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual void val2str(string &) const;
  virtual void str2val(const string &) const;

  virtual int  compare(const TSomeValue &v) const =0;
  virtual bool compatible(const TSomeValue &v) const =0;
  virtual bool operator ==(const TSomeValue &v) const;
  virtual bool operator !=(const TSomeValue &v) const;
};

WRAPPER(SomeValue)

/* TValue stores an attribute value. The object provides space for an int and float
   value, which suffice for most of uses (i.e. discrete and continuous attributes).
   For other types of attributes that would need more space, there's a pointer to
   an object of a type derived from TSomeValue.

   This approach is used instead of deriving new classes from TValue. The reason
   is in efficient implementation of TExample. TExample stores a vector of TValue's,
   and not pointers to TValue. This prohibits it from storing values of classes,
   derived from TValue.
*/

class ORANGE_API TValue {
public:
  /* valueType determines whether the value is regular (known) or not (valueDC, valueDK or some other)
     varType determines the type of the value (discrete, continuous...)

     If valueType==valueRegular, then varType determines how the value is stored
       INTVAR
         intV stores a discrete value
         sval can contain discrete PDistribution;
       FLOATVAR
         floatV stores a continuous value
         sval can contain a continuous PDistribution
       other (eg STRINGVAR?)
         sval stores some PSomeValue

     If valueType==valueDC or valueDK
       intV and floatV are not used
       sval can contain discrete distribution, continuous distribution or
         something else, depending on value of varType
  */

  enum {NONE=0, INTVAR, FLOATVAR};
  unsigned char varType;

  #define valueRegular 0
  #define valueDC 1
  #define valueDK 2

  unsigned char valueType;

  int intV;
  float floatV;
  PSomeValue svalV;

  TValue()
  : varType(NONE),
    valueType(numeric_limits<char>::max()),
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN())
  {}

  explicit TValue(const int &v)
  : varType(INTVAR),
    valueType(valueRegular),
    intV(v),
    floatV(numeric_limits<float>::quiet_NaN())
  {}

  explicit TValue(const int &anintV, PSomeValue v)
  : varType(INTVAR),
    valueType(valueRegular),
    intV(anintV),
    floatV(numeric_limits<float>::quiet_NaN()),
    svalV(v)
  {}

  explicit TValue(const float &v)
  : varType(FLOATVAR),
    valueType(valueRegular),
    intV(numeric_limits<int>::max()),
    floatV(v)
  {}

  explicit TValue(const double &v)
  : varType(FLOATVAR),
    valueType(valueRegular),
    intV(numeric_limits<int>::max()),
    floatV(float(v))
  {}

  explicit TValue(const float &afloatV, PSomeValue v)
  : varType(FLOATVAR),
    valueType(valueRegular),
    intV(numeric_limits<int>::max()),
    floatV(afloatV),
    svalV(v)
  {}

  explicit TValue(PSomeValue v, const unsigned char &t, const signed char &spec = valueRegular)
  : varType(t),
    valueType(spec),
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN()),
    svalV(v)
  {}

  explicit TValue(const unsigned char &t, signed char spec=valueDC)
  : varType(t),
    valueType(spec),
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN())
  { if (!spec)
      raiseErrorWho("Value", "illegal 'valueType' for special value");
  }

  TValue(const TValue &other)
  { *this = other; }

  inline operator int() const
  { return (!valueType && (varType==INTVAR)) ? intV : numeric_limits<int>::max(); }

  inline operator float() const
  { return (!valueType && (varType==FLOATVAR)) ? floatV : numeric_limits<float>::signaling_NaN(); }

  inline void killValues()
  { intV = numeric_limits<int>::max();
    floatV = numeric_limits<float>::quiet_NaN();
  }

  inline void setDC()
  { killValues();
    valueType = valueDC;
  }

  inline void setDK()
  { killValues();
    valueType = valueDK;
  }

  inline void setSpecial(int spec)
  { killValues();
    valueType = (unsigned char)spec;
  }

  inline bool isRegular() const
  { return valueType==valueRegular; }

  inline bool isDC() const
  { return valueType==valueDC; }

  inline bool isDK() const
  { return valueType==valueDK; }

  inline bool isSpecial() const
  { return valueType>0; }

  inline TValue &operator =(const TValue &other)
  { varType = other.varType;
    valueType = other.valueType;
    intV = other.intV;
    floatV = other.floatV;
    svalV = CLONE(TSomeValue, other.svalV);
    return *this;
  }

  #define CASES(in,fl,sv)             \
    {                                 \
      switch(varType) {               \
        case INTVAR:     return in;   \
        case FLOATVAR:   return fl;   \
        default:   return sv;         \
      }                               \
    }


  inline int compare(const TValue &v) const
  {
    if (isSpecial())
      return v.isSpecial() ? 0 : 1;
    else
      if (v.isSpecial())
        return -1;

    CASES((sign(intV-v.intV)), (sign(floatV-v.floatV)), (svalV->compare(v.svalV.getReference())))
  }


  inline bool compatible(const TValue &v) const
  {
    if (isSpecial() || v.isSpecial())
      return true;

    CASES((intV==v.intV), (floatV==v.floatV), (svalV->compatible(v.svalV.getReference())));
  }


  inline bool operator ==(const TValue &v) const
  {
    if (isSpecial())
      return v.isSpecial();
    else
      if (v.isSpecial())
        return false;

    CASES((intV==v.intV), (floatV==v.floatV), (svalV->operator == (v.svalV.getReference())));
  }


  inline bool operator !=(const TValue &v) const
  {
    if (isSpecial())
      return !v.isSpecial();
    else
      if (v.isSpecial())
        return true;

    CASES((intV!=v.intV), (floatV!=v.floatV), (svalV->operator != (v.svalV.getReference())))
  }
};


inline void intValInit(TValue &val, const int &i, const int &vt = valueRegular)
{
  val.varType = TValue::INTVAR;
  val.valueType = vt;
  val.intV = i;
  val.svalV = PSomeValue();
}


inline void floatValInit(TValue &val, const float &f, const int &vt = valueRegular)
{
  val.varType = TValue::FLOATVAR;
  val.valueType = vt;
  val.floatV = f;
  val.svalV = PSomeValue();
}


inline bool mergeTwoValues(TValue &mergedValue, const TValue &newValue, bool alreadyDefined)
{
  if (alreadyDefined)
    if (mergedValue.isSpecial()) {
      if (newValue.isSpecial())
        return mergedValue.valueType == newValue.valueType;
    }
    else
      return newValue.isSpecial() || (mergedValue == newValue);

  mergedValue = newValue;
  return true;
}

#endif

