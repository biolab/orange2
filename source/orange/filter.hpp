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


class TValueRange : public TOrange {
public:
  float min, max; // would use union but can't since PDiscDistribution has constructors...
  PDiscDistribution probs;

  signed char special;

  TValueRange(float min=0.0, float max=0.0, signed char =-1);
  TValueRange(PDiscDistribution, signed char =-1);
};

WRAPPER(ValueRange)


/// With given probability selects examples for which any of the given attribute has some of the given values 
class TFilter_sameValues : public TFilter {
public:
  __REGISTER_CLASS

  VECTOR_INTERFACE(PValueRange, values)

  /*  Discrete value with index i is `matching' if random number from [0,1) is lower than probs[i][j],
      i is the attribute's index. For continuous attributes, value is `matching' if it is from the
      specified interval [min, max]; if min>max the attribute's value is required to be outside the
      interval [min, max]. Special values are matching if special is 1 and not if it is 0; if
      special is -1, special values are simply ignored. Example if chosen if the test succeds
      for all the specified values (if doAnd) or for any of them (if !doAnd). 
      If no values could be verified (all special and ignored, or something like that), the example
      is chosen if doAnd and not if !doAnd. 
      Negating is applied to the return value (on the final result, not on each term of con/disjunction).
      */

  /* Decides whether it selects examples with all values being appropriate (doAnd==true)
      or with at least one being appropriate */
  bool doAnd; //P if true, filter computes conjunction, otherwise disjunction

  TFilter_sameValues(bool anAnd=true, bool negate=false, PDomain =PDomain());
  TFilter_sameValues(const vector<PValueRange> &, bool anAnd, bool=false, PDomain =PDomain());
  virtual bool operator()(const TExample &);

  int traverse(visitproc visit, void *arg);
  int dropReferences();
};


WRAPPER(Filter_sameValues)

class TValueFilter : public TFilter_sameValues {
public:
  __REGISTER_CLASS

  TValueFilter(bool anAnd, bool aneg=false, PDomain = PDomain());
  TValueFilter(const TMultiStringParameters &pars, string keyword, PDomain dom, bool anAnd, bool aneg=false);
  TValueFilter(istream &istr, PDomain dom, bool anAnd, bool aneg=false);

  void decode(const TMultiStringParameters &pars, string keyword);
  void decode(istream &istr);
  void decode(const vector<string> &drops);
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
