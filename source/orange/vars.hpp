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


#ifndef __VARS_HPP
#define __VARS_HPP

#include <vector>
#include <string>
#include "orvector.hpp"

using namespace std;

#include "garbage.hpp"
#include "root.hpp"

class TValue;
class TExample;
class TTransformValue;
class TDistribution;

WRAPPER(Domain);
WRAPPER(ExampleGenerator)
WRAPPER(Classifier);
WRAPPER(Learner);


#include "getarg.hpp"
#include "stladdon.hpp"

#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267)
#endif

WRAPPER(Variable);
WRAPPER(Classifier)

class TVariable : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  int  varType; //P variable type
  bool ordered; //P variable values are ordered
  bool distributed; //P variable values are distributions

  PVariable sourceVariable; //P The attribute that this attribute is computed from (when applicable)
  PClassifier getValueFrom; //P Function to compute the value from values of other variables
  bool getValueFromLocked;

  TVariable(const int &avarType=TValue::NONE, const bool &ordered = false);
  TVariable(const string &aname, const int &avarType=TValue::NONE, const bool &ordered = false);

  virtual TValue DC() const;
  virtual TValue DK() const;
  virtual TValue specialValue(int) const;

  bool str2special(const string &valname, TValue &valu) const;  // Those two functions convert ? to DK, ~ to DC and vice-versa
  bool special2str(const TValue &val, string &str) const;

  virtual void val2str(const TValue &val, string &str) const =0;
  virtual void str2val(const string &valname, TValue &valu) const =0;
  virtual bool str2val_try(const string &valname, TValue &valu) const;
  virtual void str2val_add(const string &valname, TValue &valu);

  virtual bool   firstValue(TValue &val) const =0;
  virtual bool   nextValue(TValue &val) const =0;
  virtual TValue randomValue(const int &rand=-1) =0;

  // Returns the number of different values, -1 if it cannot be done (for example, if variable is continuous)
  virtual int  noOfValues() const =0;

  virtual TValue computeValue(const TExample &);
};

WRAPPER(Variable)

#define TVarList TOrangeVector<PVariable> 
VWRAPPER(VarList)

#define TVarListList TOrangeVector<PVarList> 
VWRAPPER(VarListList)

//WRAPPERNML(IdList);

class TEnumVariable : public TVariable {
public:
  __REGISTER_CLASS

  PStringList values; //P attribute's values
  int baseValue;      //P the index of the base value

  TEnumVariable();
  TEnumVariable(const string &aname);
  TEnumVariable(const string &aname, PStringList val);
  TEnumVariable(const TEnumVariable &);

  void addValue(const string &);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  virtual int  noOfValues() const;

  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu) const;
  virtual bool str2val_try(const string &valname, TValue &valu) const;
  virtual void str2val_add(const string &valname, TValue &valu);
};


// A class describing integer variables
class TIntVariable : public TVariable {
public:
  __REGISTER_CLASS

  int startValue; //P lowest value
  int endValue;   //P highest value

  TIntVariable();
  TIntVariable(const string &aname);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  virtual int noOfValues() const;

  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu) const;
  virtual bool str2val_try(const string &valname, TValue &valu) const;
};



class TFloatVariable : public TVariable {
public:
  __REGISTER_CLASS

  float startValue; //P lowest value
  float endValue;   //P highest value
  float stepValue;  //P step
  
  TFloatVariable();
  TFloatVariable(const string &aname);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  virtual int  noOfValues() const;
 
  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu) const;
  virtual bool str2val_try(const string &valname, TValue &valu) const;
};

#endif
