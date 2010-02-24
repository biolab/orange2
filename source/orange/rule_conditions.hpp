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

#ifndef __RULE_CONDITIONS_HPP
#define __RULE_CONDITIONS_HPP

#include <vector>
using namespace std;

#include "examples.hpp"

/*~***************************************************************************************
Some general templates for conditions, conjunctions and disjunctions of them
*****************************************************************************************/

template<class T>
class TCondition {
public:
  virtual bool operator()(T example)=0;

  virtual ~TCondition()
  {}
};


template<class T>
class TConditionsConjunction : public TCondition<T>, public vector<TCondition<T> *> {
public:
  typedef typename vector<TCondition<T> *>::iterator iterator;

  virtual bool operator()(T example) {
    if (!size()) return true;
    iterator ci(begin()), ce(end());
    for( ; (ci!=ce) && (**ci)(example); ci++);
    return (ci==ce);
  }
};


template<class T>
class TConditionsDisjunction : public TCondition<T>, public vector<TCondition<T> *> {
public:
  typedef typename vector<TCondition<T> *>::iterator iterator;

  virtual bool operator()(T example) {
    if (!size()) return true;
    iterator ci(begin()), ce(end());
    for( ; (ci!=ce) && !(**ci)(example); ci++);
    return (ci!=ce);
  }
};



/*~***************************************************************************************
Conditions for TAssociationRules
*****************************************************************************************/

WRAPPER(AssociationRule);

class TRuleCondAtom : public TCondition<PExample> {
public:
  int attrIndex, valueIndex;

  TRuleCondAtom(int ana, int anv=-1);
  virtual bool operator()(PExample example);
};


class TRuleCondOneOf : public TCondition<PExample> {
public:
  vector<int> attrIndices;

  TRuleCondOneOf(const vector<int> &attrs);
  bool operator()(PExample example);
};


class TRuleCondCounted : public TCondition<PAssociationRule> {
public:
  char sign;
  int occurences;
  char side;

  vector<TCondition<PExample> *> atomConditions;

  TRuleCondCounted(char as, int ano, char aside);
  TRuleCondCounted(PDomain domain, istream &istr, const vector<pair<string, vector<int> > > &sets);

  int count(PExample example);
  virtual bool operator()(PAssociationRule asr);
  bool readConditionAtoms(istream &str, vector<string> &atoms);
};


class TRuleCondConjunctions : public TConditionsConjunction<PAssociationRule> {
public:
  TRuleCondConjunctions();
  TRuleCondConjunctions(PDomain domain, istream &istr, const vector<pair<string, vector<int> > > &sets);
};


class TRuleCondDisjunctions : public TConditionsDisjunction<PAssociationRule> {
public:
  typedef vector<pair<string, vector<int> > > TSets;
  TSets sets;

  TRuleCondDisjunctions();
  TRuleCondDisjunctions(PDomain domain, istream &istr);

  void readSets(PDomain domain, istream &istr);
  void readConjunctions(PDomain domain, istream &istr);

  bool readSetAtoms(istream &str, vector<string> &atoms);
};


TRuleCondConjunctions *conditionForClassifier(const int &attributes);


#endif
