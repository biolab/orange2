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


#ifndef __LIB_KERNEL_HPP
#define __LIB_KERNEL_HPP

#include "vars.hpp"
#include "domain.hpp"
#include "examples.hpp"
#include "examplegen.hpp"
#include "table.hpp"

TExampleTable *readListOfExamples(PyObject *args);
TExampleTable *readListOfExamples(PyObject *args, PDomain, bool filterMetas = false);
PExampleGenerator exampleGenFromArgs(PyObject *args, int &weightID);
PExampleGenerator exampleGenFromArgs(PyObject *args);
PExampleGenerator exampleGenFromParsedArgs(PyObject *args);
bool varListFromDomain(PyObject *boundList, PDomain domain, TVarList &boundSet, bool allowSingle=true, bool checkForIncludance=true);
bool varListFromVarList(PyObject *boundList, PVarList varlist, TVarList &boundSet, bool allowSingle = true, bool checkForIncludance = true);
PVariable varFromArg_byDomain(PyObject *obj, PDomain domain=PDomain(), bool checkForIncludance = false);
PVariable varFromArg_byVarList(PyObject *obj, PVarList varlist, bool checkForIncludance = false);
bool convertFromPythonWithVariable(PyObject *obj, string &str);
bool varNumFromVarDom(PyObject *pyvar, PDomain domain, int &);


bool convertFromPythonWithML(PyObject *obj, string &str, const TOrangeType &base);

inline bool exampleGenFromParsedArgs(PyObject *args, PExampleGenerator &gen)
{ gen = exampleGenFromParsedArgs(args);
  return bool(gen);
}

int pt_ExampleGenerator(PyObject *args, void *egen);

typedef int (*converter)(PyObject *, void *);
converter ptd_ExampleGenerator(PDomain domain);

bool weightFromArg_byDomain(PyObject *pyweight, PDomain domain, int &weightID);
converter pt_weightByGen(PExampleGenerator &peg);

int pt_DomainContingency(PyObject *args, void *egen);




class TValueD {
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

  PSomeValue svalV;
  union {
    int intV;
    double floatV;
  };

  TValueD()
  : varType(NONE), 
    valueType(numeric_limits<char>::max()),
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN())
  {}

  explicit TValueD(const int &v)
  : varType(INTVAR),
    valueType(valueRegular),
    intV(v),
    floatV(numeric_limits<float>::quiet_NaN())
  {}

  explicit TValueD(const int &anintV, PSomeValue v)
  : varType(INTVAR), 
    valueType(valueRegular), 
    intV(anintV),
    floatV(numeric_limits<float>::quiet_NaN()),
    svalV(v)
  {}

  explicit TValueD(const float &v)
  : varType(FLOATVAR), 
    valueType(valueRegular),
    intV(numeric_limits<int>::max()),
    floatV(v)
  {}

  explicit TValueD(const double &v)
  : varType(FLOATVAR), 
    valueType(valueRegular),
    intV(numeric_limits<int>::max()),
    floatV(float(v))
  {}

  explicit TValueD(const float &afloatV, PSomeValue v)
  : varType(FLOATVAR), 
    valueType(valueRegular), 
    intV(numeric_limits<int>::max()),
    floatV(afloatV),
    svalV(v)
  {}

  explicit TValueD(PSomeValue v, const unsigned char &t, const signed char &spec = valueRegular)
  : varType(t), 
    valueType(spec), 
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN()),
    svalV(v)
  {}

  explicit TValueD(const unsigned char &t, signed char spec=valueDC)
  : varType(t),
    valueType(spec),
    intV(numeric_limits<int>::max()),
    floatV(numeric_limits<float>::quiet_NaN())
  { if (!spec)
      raiseErrorWho("Value", "illegal 'valueType' for special value"); 
  }

  TValueD(const TValue &other)
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

  inline TValueD &operator =(const TValue &other)
  { varType = other.varType;
    valueType = other.valueType;
    intV = other.intV;
    floatV = other.floatV;
    svalV = CLONE(TSomeValue, other.svalV);
    return *this;
  }
   
  #define CASES(in,fl,sv,def)         \
    { if (isSpecial() || v.isSpecial()) return def; \
      switch(varType) {               \
        case INTVAR:     return in;   \
        case FLOATVAR:   return fl;   \
        default:   return sv;   \
      }                               \
      return (def);                   \
    }


  inline int compare(const TValue &v) const
  { if (isSpecial())
      return v.isSpecial() ? 0 : 1;
    // -1 in cases now corresponds to this being regular and v special
    CASES((sign(intV-v.intV)), (sign(floatV-v.floatV)), (svalV->compare(v.svalV.getReference())), -1)
  }

  inline bool compatible(const TValue &v) const
  CASES((intV==v.intV), (floatV==v.floatV), (svalV->compatible(v.svalV.getReference())), true)

  inline signed char operator ==(const TValue &v) const
  CASES((intV==v.intV), (floatV==v.floatV), (svalV->operator == (v.svalV.getReference())), -1)

  inline signed char operator !=(const TValue &v) const
  CASES((intV!=v.intV), (floatV!=v.floatV), (svalV->operator != (v.svalV.getReference())), -1)
};


#endif
