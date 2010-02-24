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


#ifndef __FILTER_HPP
#define __FILTER_HPP

#include "examples.hpp"
#include "distvars.hpp"
#include "trindex.hpp"


WRAPPER(Filter);
/*  An abstract class, used to select examples.
    It defines an abstract bool operator()(TExample &) which must be redefined in derived classes.  */
class ORANGE_API TFilter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool negate; //P if true, filter output should be negated.
  PDomain domain; //P domain to which the examples are converted (if needed)

  TFilter(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &)=0;
  virtual PFilter deepCopy() const;
  virtual void reset();
};


#define TFilterList TOrangeVector<PFilter>
VWRAPPER(FilterList)


/// Randomly chooses examples with given probability.
class ORANGE_API TFilter_random : public TFilter {
public:
  __REGISTER_CLASS

  float prob; //P probability of selecting an example
  PRandomGenerator randomGenerator; //P random generator

  TFilter_random(const float =0.0, bool=false, PRandomGenerator = PRandomGenerator());

  virtual bool operator()(const TExample &);
};


/// Selects examples with (or without) special values. The results can be negated by setting the negate flag.
class ORANGE_API TFilter_hasSpecial : public TFilter {
public:
  __REGISTER_CLASS

  TFilter_hasSpecial(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples with all values defined
class ORANGE_API TFilter_isDefined : public TFilter {
public:
  __REGISTER_CLASS

  PAttributedBoolList check; //P tells which attributes to check; checks all if the list is empty

  TFilter_isDefined(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);

  void afterSet(const char *name);
};


/// Selects examples with all values defined
class ORANGE_API TFilter_hasMeta: public TFilter {
public:
  __REGISTER_CLASS

  int id; //P meta attribute id

  TFilter_hasMeta(const int &anid = 0, bool = false, PDomain = PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples with known class value
class ORANGE_API TFilter_hasClassValue : public TFilter {
public:
  __REGISTER_CLASS

  TFilter_hasClassValue(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples for which the attribute at position 'position' (not) equals 'value'. The result can be negated by setting the negate flas.
class ORANGE_API TFilter_sameValue : public TFilter {
public:
  __REGISTER_CLASS

  int position; //P position of the observed attribute
  TValue value; //P value that the selected examples should have

  TFilter_sameValue(const TValue & =TValue(), int=-1, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};

WRAPPER(ValueFilter)
class ORANGE_API TValueFilter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  CLASSCONSTANTS(Operator) enum { None, Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, Between, Outside, Contains, NotContains, BeginsWith, EndsWith, Listed };

  int position; //P attribute's position in domain
  int acceptSpecial; //P tells whether a special value (DK, DC...) is accepted (1), rejected (0) or ignored (-1)

  TValueFilter(const int &pos = ILLEGAL_INT, const int & = 0);
  virtual int operator()(const TExample &) const = 0; // Returns 1 for accept, 0 for reject, -1 for ignore
  virtual PValueFilter deepCopy() const;
};



class ORANGE_API TValueFilter_continuous : public TValueFilter {
public:
  __REGISTER_CLASS

  float min; //P (+ref) reference value (lower bound for interval operators)
  float max; //P upper bound for interval operators
  bool outside; //P obsolete: if true, the filter accepts the values outside the interval, not inside
  int oper; //P(&ValueFilter_Operator) operator

  TValueFilter_continuous();
  TValueFilter_continuous(const int &pos, const float &min=0.0, const float &max=0.0, const bool &outs = false, const int &accs = 0);
  TValueFilter_continuous(const int &pos, const int &op, const float &min=0.0, const float &max=0.0, const int &accs = 0);
  virtual int operator()(const TExample &) const;
  virtual PValueFilter deepCopy() const;
};

class ORANGE_API TValueFilter_discrete : public TValueFilter {
public:
  __REGISTER_CLASS

  PValueList values; //P accepted values
  bool negate; //P negate

  TValueFilter_discrete(const int &pos = ILLEGAL_INT, PValueList = PValueList(), const int &accs = 0, bool negate = false);
  TValueFilter_discrete(const int &pos, PVariable, const int &accs = 0, bool negate = false);
  virtual int operator()(const TExample &) const;
  virtual PValueFilter deepCopy() const;
};


class ORANGE_API TValueFilter_string : public TValueFilter {
public:
  __REGISTER_CLASS

  string min; //P (+ref) reference value (lower bound for interval operators)
  string max; //P upper bound for interval operators
  int oper;   //P(&ValueFilter_Operator) operator
  bool caseSensitive; //P if true (default), the operator is case sensitive

  TValueFilter_string();
  TValueFilter_string(const int &pos, const int &op, const string &min, const string &max, const int &accs = 0, const bool csens = true);
  virtual int operator()(const TExample &) const;
};


class ORANGE_API TValueFilter_stringList : public TValueFilter {
public:
  __REGISTER_CLASS

  PStringList values; //P accepted values
  bool caseSensitive; //P if true (default), the comparison is case sensitive

  TValueFilter_stringList();
  TValueFilter_stringList(const int &pos, PStringList, const int &accs = 0, const int &op = Equal, const bool csens = true);
  virtual int operator()(const TExample &) const;
};


#define TValueFilterList TOrangeVector<PValueFilter>
VWRAPPER(ValueFilterList)


/// With given probability selects examples for which any of the given attribute has some of the given values 
class ORANGE_API TFilter_values : public TFilter {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(Operator: None; Equal; NotEqual; Less; LessEqual; Greater; GreaterEqual; Between; Outside; Contains; NotContains; BeginsWith; EndsWith; Listed)

  PValueFilterList conditions; //P a list of filters

  /*  If conjunction == true, example is chosen if no values are rejected
      If conjunction == false, example is chosen if at least one value is accepted
      The above rules apply also when no values could be tested (think how :)
      
      negate is applied to whole expression, not to individual terms */

  bool conjunction; //P if true, filter computes conjunction, otherwise disjunction

  TFilter_values(bool anAnd=true, bool aneg = false, PDomain =PDomain());
  TFilter_values(PValueFilterList, bool anAnd, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
  virtual PFilter deepCopy() const;

  TValueFilterList::iterator findCondition(PVariable var, const int &varType, int &position);
  void updateCondition(PVariable var, const int &varType, PValueFilter filter);

  void addCondition(PVariable var, const TValue &val, bool negate = false);
  void addCondition(PVariable var, PValueList, bool negate = false);
  void addCondition(PVariable var, const int &oper, const float &min, const float &max);
  void addCondition(PVariable var, const int &oper, const string &min, const string &maxs);
  void addCondition(PVariable var, PStringList);
  void removeCondition(PVariable var);
};

/// Selects examples (not) equal to the given example.
class ORANGE_API TFilter_sameExample : public TFilter {
public:
  __REGISTER_CLASS

  PExample example; //P example with which examples are compared

  TFilter_sameExample(PExample, bool=false);
  virtual bool operator()(const TExample &);
};


/// Selects examples (not) compatible with the given example.
class ORANGE_API TFilter_compatibleExample : public TFilter {
public:
  __REGISTER_CLASS

  PExample example; //P example with which examples are compared

  TFilter_compatibleExample(PExample, bool=false);
  virtual bool operator()(const TExample &);
};


class ORANGE_API TFilter_conjunction : public TFilter {
public:
  __REGISTER_CLASS

  PFilterList filters; //P a list of filters;

  TFilter_conjunction();
  TFilter_conjunction(PFilterList);
  virtual bool operator()(const TExample &);
};


class ORANGE_API TFilter_disjunction : public TFilter {
public:
  __REGISTER_CLASS

  PFilterList filters; //P a list of filters;

  TFilter_disjunction();
  TFilter_disjunction(PFilterList);
  virtual bool operator()(const TExample &);
};

#endif
