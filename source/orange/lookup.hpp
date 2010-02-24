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


#ifndef __LOOKUP_HPP
#define __LOOKUP_HPP

#include "classify.hpp"
#include "vars.hpp"
#include "learn.hpp"

WRAPPER(ExampleGenerator);
WRAPPER(EFMDataDescription)

class TExampleTable; 

class ORANGE_API TClassifierByLookupTable : public TClassifier {
public:
  __REGISTER_ABSTRACT_CLASS

  TClassifierByLookupTable(PVariable, PValueList);

  PValueList lookupTable; //PR a list of class values, one for each attribute value
  PDistributionList distributions; //PR a list of class distributions, one for each attribute value

  virtual int getIndex(const TExample &ex, TExample *conv=NULL) = 0;
  virtual void giveBoundSet(TVarList &boundSet) = 0;

  void valuesFromDistributions();
};


class ORANGE_API TClassifierByLookupTable1 : public TClassifierByLookupTable {
public:
  __REGISTER_CLASS

  PVariable variable1; //PR(+variable) attribute used for classification

  TClassifierByLookupTable1(PVariable aclass, PVariable avar);

  virtual TValue operator()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist);

  void setLastDomain(PDomain domain);
  void replaceDKs(TDiscDistribution &valDistribution);
  int getIndex(const TExample &ex, TExample *conv=NULL);
  void giveBoundSet(TVarList &boundSet);

private:
  long    lastDomainVersion;
  int     lastVarIndex;
};


WRAPPER(ProbabilityEstimator);

class ORANGE_API TClassifierByLookupTable2 : public TClassifierByLookupTable {
public:
  __REGISTER_CLASS

  PVariable variable1; //PR the first attribute used for classification
  PVariable variable2; //PR the second attribute used for classification
  int noOfValues1; //PR number of values of the first attribute
  int noOfValues2; //PR number of values of the second attribute
  PEFMDataDescription dataDescription; //P data description

  TClassifierByLookupTable2(PVariable aclass, PVariable, PVariable, PEFMDataDescription =PEFMDataDescription());

  virtual TValue operator()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist);

  void setLastDomain(PDomain domain);
  int getIndex(const TExample &ex, TExample *conv=NULL);
  void replaceDKs(PExampleGenerator examples, bool useBayes=true);
  void giveBoundSet(TVarList &boundSet);

private:
  long    lastDomainVersion;
  int     lastVarIndex1, lastVarIndex2;
};


class ORANGE_API TClassifierByLookupTable3 : public TClassifierByLookupTable {
public:
  __REGISTER_CLASS

  PVariable variable1; //PR the first attribute used for classification
  PVariable variable2; //PR the second attribute used for classification
  PVariable variable3; //PR the third attribute used for classification
  int noOfValues1; //PR number of values of the first attribute
  int noOfValues2; //PR number of values of the second attribute
  int noOfValues3; //PR number of values of the third attribute
  PEFMDataDescription dataDescription; //P data description

  TClassifierByLookupTable3(PVariable aclass, PVariable, PVariable, PVariable, PEFMDataDescription =PEFMDataDescription());

  virtual TValue operator()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist);

  void setLastDomain(PDomain domain);
  int  getIndex(const TExample &ex, TExample *conv=NULL);
  void replaceDKs(PExampleGenerator examples, bool useBayes=true);
  void giveBoundSet(TVarList &boundSet);

private:
  long    lastDomainVersion;
  int     lastVarIndex1, lastVarIndex2, lastVarIndex3;
};


class ORANGE_API TClassifierByLookupTableN : public TClassifierByLookupTable {
public:
  __REGISTER_CLASS

  PVarList variables; //PR attributes
  PIntList noOfValues; //PR number of values for each attribute
  PEFMDataDescription dataDescription; //P data description

  TClassifierByLookupTableN(PVariable aclass, PVarList avars, PEFMDataDescription =PEFMDataDescription());

  virtual TValue operator()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &example, TValue &value, PDistribution &dist);

  void setLastDomain(PDomain domain);
  int  getIndex(const TExample &ex, TExample *conv=NULL);
  void replaceDKs(PExampleGenerator examples, bool useBayes=true);
  void giveBoundSet(TVarList &boundSet);

private:
  long    lastDomainVersion;
  vector<int> lastVarIndices;
};


WRAPPER(DomainContingency);
WRAPPER(ProbabilityEstimator);

class ORANGE_API TLookupLearner : public TLearner {
public:
  __REGISTER_CLASS

  enum {UnknownsIgnore = 0, UnknownsDistribute, UnknownsKeep};

  PLearner learnerForUnknown; //P a learner for classifying cases not found in the table
  bool allowFastLookups; //P if true, it constructs LookupClassifiers for <=3 attributes
  int unknownsHandling; //P 0 omit examples with unknowns, 1 distribute them, 2 keep them in table
  
  TLookupLearner();

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};

WRAPPER(ExampleTable);


class ORANGE_API TClassifierByExampleTable : public TClassifierFD {
public:
  __REGISTER_CLASS

  PExampleTable sortedExamples; //P a table of examples
  bool containsUnknowns; //P if true, the table contains unknown values
  PClassifier classifierForUnknown;  //P a classifier for unknown cases
  PEFMDataDescription dataDescription; //P data description

  TClassifierByExampleTable(PDomain dom = PDomain());
  TClassifierByExampleTable(PExampleGenerator, PClassifier = PClassifier());

  void afterSet(const char *name);

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &ex, TValue &pred, PDistribution &dist);
  
  PDistribution classDistributionLow(const TExample &);
};

#endif
