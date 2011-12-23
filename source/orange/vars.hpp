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


#ifndef __VARS_HPP
#define __VARS_HPP

#include <string>
#include <list>
#include <set>
#include <map>
#include "orvector.hpp"

using namespace std;

#include "root.hpp"

class ORANGE_API TValue;
class ORANGE_API TExample;
class ORANGE_API TTransformValue;
class ORANGE_API TDistribution;

WRAPPER(Domain);
WRAPPER(ExampleGenerator)
WRAPPER(Classifier);
WRAPPER(RandomGenerator)

//#include "getarg.hpp"
#include "stladdon.hpp"

#ifdef _MSC_VER
 #pragma warning (disable : 4786 4114 4018 4267 4251)
#endif

WRAPPER(Variable);
WRAPPER(Classifier)

class MMV;

class ORANGE_API TVariable : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  CLASSCONSTANTS(Type: None=(int)TValue::NONE; Discrete=(int)TValue::INTVAR; Continuous=(int)TValue::FLOATVAR; Other=(int)(TValue::FLOATVAR+1); String=(int)STRINGVAR);
  
  //string name; //variable's name - hiding it!

  int  varType; //P(&Variable_Type) variable type
  bool ordered; //P variable values are ordered
  bool distributed; //P variable values are distributions
  int defaultMetaId; //P default (proposed, suggested...) meta id for this variable

  PRandomGenerator randomGenerator; //P random generator for random values (initialized when first needed)

  PVariable sourceVariable; //P The attribute that this attribute is computed from (when applicable)
  PClassifier getValueFrom; //P Function to compute the value from values of other variables
  bool getValueFromLocked;

protected:
  TValue DC_value;
  TValue DK_value;

private:
  string name;

  void registerVariable();
  void removeVariable();

public:

  string get_name() const {
    return name;
  }

  void set_name(const string &a) {
    //update the map
    removeVariable();
    name = a;
    registerVariable();
  }

  static MMV allVariablesMap;

  ~TVariable();

  /* Status codes for getExisting and make. The codes refer to the difference between
     the requested and the existing variable.
     OK                  the new variable contains at least one of the existing values,
                         and no new values; there is no problem with their order
     MissingValues       the new variable contains at least one of the existing values,
                         and some new oness; there is no problem with their order
     NoRecognizedValues  the new variable contains no existing values
     Incompatible        the new variable prescribes an order of values which is
                         incompatible with the existing
     NotFound            the variable with that name and type doesn't exist yet
  */
  CLASSCONSTANTS(MakeStatus) enum { OK, MissingValues, NoRecognizedValues, Incompatible, NotFound };
  
  /* This will search for an existing variable and return it unless the status (above)
     equals or exceeds the failOn argument, Incompatible or NotFound.
     The status is return if status!=NULL */
  static TVariable *getExisting(const string &name, const int &varType, TStringList *fixedOrderValues = NULL, set<string> *values = NULL,
                                const int failOn = Incompatible, int *status = NULL);
                                
  /* Gets an existing variable or makes a new one. A new one is made if there is no
     existing variable by that name or its status (above) equals or exceeds createNewOne.
     The returned status equals to the result of the search for an existing variable,
     except if createNewOn==OK, in which case status is always OK.  */
  static TVariable *make(const string &name, const int &varType, TStringList *fixedOrderValues = NULL, set<string> *value = NULL,
                                const int createNewOn = Incompatible, int *status = NULL);
                  
  virtual bool isEquivalentTo(const TVariable &old) const;
  
  TVariable(const int &avarType = TValue::NONE, const bool &ordered = false);
  TVariable(const string &aname, const int &avarType=TValue::NONE, const bool &ordered = false);

  virtual const TValue &DC() const;
  virtual const TValue &DK() const;
  virtual TValue specialValue(int) const;

  bool str2special(const string &valname, TValue &valu) const;  // Those two functions convert ? to DK, ~ to DC and vice-versa
  bool special2str(const TValue &val, string &str) const;

  virtual void val2str(const TValue &val, string &str) const =0;
  virtual void str2val(const string &valname, TValue &valu) =0;
  virtual bool str2val_try(const string &valname, TValue &valu);
  virtual void str2val_add(const string &valname, TValue &valu);

  virtual void val2filestr(const TValue &val, string &str, const TExample &) const;
  virtual void filestr2val(const string &valname, TValue &valu, TExample &);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  // Returns the number of different values, -1 if it cannot be done (for example, if variable is continuous)
  virtual int  noOfValues() const =0;

  virtual TValue computeValue(const TExample &);
};

WRAPPER(Variable)

#define TVarList TOrangeVector<PVariable> 
VWRAPPER(VarList)

#define TVarListList TOrangeVector<PVarList> 
VWRAPPER(VarListList)

class ORANGE_API TEnumVariable : public TVariable {
public:
  __REGISTER_CLASS

  PStringList values; //P attribute's values
  int baseValue;      //P the index of the base value

  TEnumVariable();
  TEnumVariable(const string &aname);
  TEnumVariable(const string &aname, PStringList val);
  TEnumVariable(const TEnumVariable &);

  virtual bool isEquivalentTo(const TVariable &old) const;

  void addValue(const string &);
  bool hasValue(const string &);

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  virtual int  noOfValues() const;

  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu);
  virtual bool str2val_try(const string &valname, TValue &valu);
  virtual void str2val_add(const string &valname, TValue &valu);

  bool checkValuesOrder(const TStringList &refValues);
  static void presortValues(const set<string> &unsorted, vector<string> &sorted);

private:
  map<string, int> valuesTree;
  void createValuesTree();
};




class ORANGE_API TFloatVariable : public TVariable {
public:
  __REGISTER_CLASS

  float startValue; //P lowest value
  float endValue;   //P highest value
  float stepValue;  //P step
  
  int numberOfDecimals; //P number of digits after decimal point
  bool scientificFormat; //P use scientific format in output
  int adjustDecimals; //P adjust number of decimals according to the values converted (0 - no, 1 - yes, 2 - yes, but haven't seen any yet)

  TFloatVariable();
  TFloatVariable(const string &aname);

  virtual bool isEquivalentTo(const TVariable &old) const;

  virtual bool   firstValue(TValue &val) const;
  virtual bool   nextValue(TValue &val) const;
  virtual TValue randomValue(const int &rand=-1);

  virtual int  noOfValues() const;
 
  virtual void val2str(const TValue &val, string &str) const;
  virtual void str2val(const string &valname, TValue &valu);
  virtual bool str2val_try(const string &valname, TValue &valu);
  int str2val_low(const string &valname, TValue &valu);
};

#endif
