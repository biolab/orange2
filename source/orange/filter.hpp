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


#ifndef __FILTER_HPP
#define __FILTER_HPP

#include "examples.hpp"
#include "distvars.hpp"
#include "trindex.hpp"

/*  An abstract class, used to select examples.
    It defines an abstract bool operator()(TExample &) which must be redefined in derived classes.  */
class TFilter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool negate; //P if true, filter output should be negated.
  PDomain domain; //P domain to which the examples are converted (if needed)

  TFilter(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &)=0;
  virtual void reset();
};

WRAPPER(Filter);


/// Randomly chooses examples with given probability.
class TFilter_random : public TFilter {
public:
  __REGISTER_CLASS

  float prob; //P probability of selecting an example
  PRandomGenerator randomGenerator; //P random generator

  TFilter_random(const float =0.0, bool=false, PDomain =PDomain());

  virtual bool operator()(const TExample &);
};


/// Selects examples with (or without) special values. The results can be negated by setting the negate flag.
class TFilter_hasSpecial : public TFilter {
public:
  __REGISTER_CLASS

  TFilter_hasSpecial(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples with known class value
class TFilter_hasClassValue : public TFilter {
public:
  __REGISTER_CLASS

  TFilter_hasClassValue(bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples for which the attribute at position 'position' (not) equals 'value'. The result can be negated by setting the negate flas.
class TFilter_sameValue : public TFilter {
public:
  __REGISTER_CLASS

  int position; //P position of the observed attribute
  TValue value; //P value that the selected examples should have

  TFilter_sameValue(const TValue & =TValue(), int=-1, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


class TValueFilter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  int acceptSpecial; //P tells whether a special value (DK, DC...) is accepted (1), rejected (0) or ignored (-1)

  TValueFilter(const int & = -1);
  virtual int operator()(const TValue &) const = 0; // Returns 1 for accept, 0 for reject, -1 for ignore
};

WRAPPER(ValueFilter)

class TValueFilter_continuous : public TValueFilter {
public:
  __REGISTER_CLASS

  float min; //P minimal acceptable value
  float max; //P maximal acceptable value
  bool outside; //P it true, the filter accepts the values outside the interval, not inside

  TValueFilter_continuous(const float &min=0.0, const float &max=0.0, const bool &neg = false, const int &accs = -1);
  virtual int operator()(const TValue &) const;
};


class TValueFilter_discrete : public TValueFilter {
public:
  __REGISTER_CLASS

  PValueList acceptableValues; //P acceptable values

  TValueFilter_discrete(PValueList = PValueList(), const int &accs = -1);
  TValueFilter_discrete(PVariable, const int &accs = -1);
  virtual int operator()(const TValue &) const;
};


#define TValueFilterList TOrangeVector<PValueFilter>
VWRAPPER(ValueFilterList)


/// With given probability selects examples for which any of the given attribute has some of the given values 
class TFilter_Values : public TFilter {
public:
  __REGISTER_CLASS

  PValueFilterList values; //P a list of filters

  /*  If doAnd == true, example is chosen if no values are rejected
      If doAnd == false, example is chosen if at least one value is accepted
      The above rules apply also when no values could be tested (think how :)
      
      negate is applied to whole expression, not to individual terms */

  bool doAnd; //P if true, filter computes conjunction, otherwise disjunction

  TFilter_Values(bool anAnd=true, bool aneg = false, PDomain =PDomain());
  TFilter_Values(PValueFilterList, bool anAnd, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
};


/// Selects examples (not) equal to the given example.
class TFilter_sameExample : public TFilter {
public:
  __REGISTER_CLASS

  PExample example; //P example with which examples are compared

  TFilter_sameExample(PExample, bool=false);
  virtual bool operator()(const TExample &);
};


/// Selects examples (not) compatible with the given example.
class TFilter_compatibleExample : public TFilter {
public:
  __REGISTER_CLASS

  PExample example; //P example with which examples are compared

  TFilter_compatibleExample(PExample, bool=false);
  virtual bool operator()(const TExample &);
};


/*  Selects example for which the element of the given table equals specified value.
    Each element in the table should correspond to one example which will be passed
    to operator(), in the same order. Therefore, size of table should equal number of
    invokations of the operator. If, however, table of indices is too small, it is
    reused from the beginning. */
class TFilter_index : public TFilter {
public:
  __REGISTER_CLASS

  PLongList indices; //P indices
  int value; //P selected value

  /// Temporary position in table of indices
  vector<FOLDINDEXTYPE>::iterator position;
  
  TFilter_index();
  TFilter_index(TFoldIndices &, int aval, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);
  virtual void reset();
};

WRAPPER(Filter_index)

#endif
