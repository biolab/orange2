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


#ifndef __LOOKUP_HPP
#define __LOOKUP_HPP

#include "classify.hpp"
#include "vars.hpp"
#include "learn.hpp"

WRAPPER(ExampleGenerator);
WRAPPER(EFMDataDescription)

class TExampleTable; 

class TClassifierByLookupTable : public TClassifier {
public:
  __REGISTER_CLASS

  PVariable variable; //P attribute used for classification
  PValueList lookupTable; //P a list of class values, one for each attribute value
  PDistributionList distributions; //P a list of class distributions, one for each attribute value

  TClassifierByLookupTable(PVariable aclass, PVariable avar);

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

class TClassifierByLookupTable2 : public TClassifier {
public:
  __REGISTER_CLASS

  PVariable variable1; //P the first attribute used for classification
  PVariable variable2; //P the second attribute used for classification
  int noOfValues1; //P number of values of the first attribute
  int noOfValues2; //P number of values of the second attribute
  PValueList lookupTable; //P a list of class values, on for each combination of attribute values
  PDistributionList distributions; //P a list of class distributions, on for each combination of attributes' values
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


class TClassifierByLookupTable3 : public TClassifier {
public:
  __REGISTER_CLASS

  PVariable variable1; //P the first attribute used for classification
  PVariable variable2; //P the second attribute used for classification
  PVariable variable3; //P the third attribute used for classification
  int noOfValues1; //P number of values of the first attribute
  int noOfValues2; //P number of values of the second attribute
  int noOfValues3; //P number of values of the third attribute
  PValueList lookupTable; //P a list of class values, on for each combination of attribute values
  PDistributionList distributions; //P a list of class distributions, on for each combination of attributes' values
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



WRAPPER(DomainContingency);
WRAPPER(PProbabilityEstimator);

class TLookupLearner : public TLearner {
public:
  __REGISTER_CLASS

  PLearner learnerForUnknown; //P a learner for classifying unknown cases

  TLookupLearner();
  TLookupLearner(const TLookupLearner &old);

  virtual PClassifier operator()(PExampleGenerator, const int & =0);
};

WRAPPER(ExampleTable);


class TClassifierByExampleTable : public TClassifierFD {
public:
  __REGISTER_CLASS

  PDomain domainWithoutClass; //PR a domain (of sortedExamples) but without the class value
  PExampleTable sortedExamples; //PR a table of examples
  PClassifier classifierForUnknown;  //P a classifier for unknown cases

  TClassifierByExampleTable(PDomain dom = PDomain());
  TClassifierByExampleTable(PExampleGenerator, PClassifier = PClassifier());

  void afterSet(const string &name);

  virtual TValue operator ()(const TExample &);
  virtual TValue operator ()(PDistribution);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &ex, TValue &pred, PDistribution &dist);
  
  PDistribution classDistributionLow(const TExample &exam);
};


WRAPPER(DomainDistributions);
#include "examplegen.hpp"

/*  Classifies by looking for the example in the example set */
class TClassifierFromGenerator : public TDefaultClassifier {
public:
  __REGISTER_CLASS

  PExampleGenerator generator; //P an example generator
  int weightID; //P an id of meta-attribute with weights
  PDomain domainWithoutClass; //P a class-less domain
  PEFMDataDescription dataDescription; //P data description
  PClassifier classifierForUnknown; //P a classifier for examples that were not found

  TClassifierFromGenerator();
  TClassifierFromGenerator(PVariable &);
  TClassifierFromGenerator(PVariable &, TValue &, TDistribution &);
  TClassifierFromGenerator(PExampleGenerator, int weightID=0);
  TClassifierFromGenerator(const TClassifierFromGenerator &old);

//  virtual int copiesOfClassVar();

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
};

#endif
