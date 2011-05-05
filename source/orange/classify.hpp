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

#ifndef _CLASSIFY_HPP
#define _CLASSIFY_HPP

#include <string>

#include "examples.hpp"
#include "distvars.hpp"

using namespace std;

WRAPPER(Classifier)

WRAPPER(EFMDataDescription);

#define TClassifierList TOrangeVector<PClassifier> 
VWRAPPER(ClassifierList)

/* Classifiers have three methods for classification.
   - operator() returns TValue
   - classDistribution return PDistribution
   - predictionAndDistribution returns both

   At least one of the first two need to be overloaded. If the method
   can return probabilities (or at least something the closely
   ressembles it), it should redefine the second.

   computesProbabilities should be set if classDistribution is overloaded.

   Here are different scenarios:

   1. Derived class overloads only operator() (returning TValue) and
      sets computeProbabilities to false.

      In this case, the inherited PDistribution will call operator()
      and construct a PDistribution in which the returned TValue will
      have a probability of 1.0. Similarly, predictionAndDistribution
      will call operator() and construct PDistribution.

   2. Derived class overloads classDistribution and sets
      computeProbabilities.

      The inherited operator() will call classDistribution and use the
      distribution's method highestProbValue() to select a value to return.
      (Note that it randomly selects one of the values (and not the first
      value) with the highest probability. This can lead to
      different classification of the same example.) Method
      predictionAndDistribution behaves similarly.

   3. operator() and classDistribution are overloaded, while
      predictionAndDistribution is not.

      computesProbabilities decides which of the two methods are used
      by the inherited predictionAndDistribution.

   4. Only predictionAndDistribution is overloaded.

      This is illegal. If prediction cannot be based on probabilities
      (for example, if there is a way to prefer one of predictions with
      the same probability), you should redefine all three methods.
      computesProbabilities can be set to whatever you like.

   Most classifiers will use scenario 2 -- return classDistributions
   and leave the rest to methods inherited from TClassifier.
*/

class ORANGE_API TClassifier : public TOrange {
public:
  __REGISTER_CLASS

  PVariable classVar; //P class variable
  bool computesProbabilities; //P set if classifier computes class probabilities (if not, it assigns 1.0 to the predicted)

  TClassifier(const bool &cp=false);
  TClassifier(const PVariable &, const bool &cp=false);
  TClassifier(const TClassifier &old);

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);

  virtual TValue operator()(const TExample &, PEFMDataDescription);
  virtual PDistribution classDistribution(const TExample &, PEFMDataDescription);
};


class ORANGE_API TClassifierFD : public TClassifier {
public:
  __REGISTER_CLASS

  PDomain domain; //P domain

  TClassifierFD(const bool &cp =false);
  TClassifierFD(PDomain, const bool &cp =false);
  TClassifierFD(const TClassifierFD &old);

  void afterSet(const char *name);
};


class ORANGE_API TDefaultClassifier : public TClassifier {
public:
  __REGISTER_CLASS

  TValue defaultVal; //P default prediction
  PDistribution defaultDistribution; //P default distribution

  TDefaultClassifier();
  TDefaultClassifier(PVariable);
  TDefaultClassifier(PVariable, PDistribution defDis);
  TDefaultClassifier(PVariable, const TValue &defVal, PDistribution defDis);
  TDefaultClassifier(const TDefaultClassifier &old);
  
  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);
  virtual void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
};


class ORANGE_API TRandomClassifier : public TClassifier {
public:
  __REGISTER_CLASS

  PDistribution probabilities; //P probabilities of predictions

  TRandomClassifier(PVariable acv=PVariable());
  TRandomClassifier(const TDistribution &probs);
  TRandomClassifier(PVariable acv, const TDistribution &probs);
  TRandomClassifier(PDistribution);
  TRandomClassifier(PVariable acv, PDistribution);

  TValue operator()(const TExample &);
  PDistribution classDistribution(const TExample &);
  void predictionAndDistribution(const TExample &, TValue &val, PDistribution &dist);
};


WRAPPER(DomainDistributions);
WRAPPER(ExampleGenerator);


#ifdef _MSC_VER
  template class ORANGE_API std::vector<float>;
#endif

class ORANGE_API TEFMDataDescription : public TOrange {
public:
  __REGISTER_CLASS

  PDomain domain; //PR domain
  PDomainDistributions domainDistributions; //C distributions of values for attributes
  std::vector<float> averages;
  vector<float> matchProbabilities; // if you intend to really export this class, you'll need to define 'afterSet' for domain distributions
  int originalWeight, missingWeight;

  TEFMDataDescription(PDomain, PDomainDistributions=PDomainDistributions(), int ow=0, int mw=0);
  void getAverages();

  float getExampleWeight(const TExample &) const;
  float getExampleMatch(const TExample &, const TExample &);
};


class ORANGE_API TExampleForMissing : public TExample {
public:
  __REGISTER_CLASS

  PEFMDataDescription dataDescription; //P data description
  vector<int> DKs;
  vector<int> DCs;

  TExampleForMissing(PDomain, PEFMDataDescription = PEFMDataDescription());
  TExampleForMissing(const TExample &orig, PEFMDataDescription =PEFMDataDescription());
  TExampleForMissing(const TExampleForMissing &orig);
  TExampleForMissing(PDomain dom, const TExample &orig, PEFMDataDescription);

  virtual TExampleForMissing &operator =(const TExampleForMissing &orig);
  virtual TExample &operator =(const TExample &orig);

  void resetExample();
  bool nextExample();
  bool hasMissing();
};


#endif
