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


#ifndef __SPEC_CONTINGENCY_HPP
#define __SPEC_CONTINGENCY_HPP

#include "transdomain.hpp"
#include "contingency.hpp"

WRAPPER(Learner)
WRAPPER(Preprocessor)

class ORANGE_API TComputeDomainContingency_DomainTransformation : public TComputeDomainContingency {
public:
  __REGISTER_CLASS

  PDomainTransformerConstructor domainTransformerConstructor; //P constructs a domain in which each attribute corresponds one of originals
  bool resultInOriginalDomain; //P tells whether the resulting DomainContingency should consist of original attributes

  // The examples are fed to a domainTransformerConstructor which returns a new domain.
  // Examples are transformed to the new domain (this may be done on the fly or not - do not
  // rely on anything regarding this).
  // If resultInOriginalDomain is false PDomainContingency will be built from the transformed examples
  // and will include the new attributes.
  // If it is true, than the domainTransformerConstructor is expected to have constructed
  // a domain in which the sourceVariable field of each new attribute points to an old attribute.
  // The new attribute should have the same set of values (in the same order!) as the corresponding
  // old attribute. The new attribute would in practice only 'correct' some values of the old
  // one (e.g. imputation, noise handling).
  // The order of the attributes in the new domain does not need to correspond to that of the original.

  virtual PDomainContingency operator()(PExampleGenerator, const long &weightID=0);
};


class ORANGE_API TComputeDomainContingency_ImputeWithClassifier : public TComputeDomainContingency {
public:
  __REGISTER_CLASS

  PLearner learnerForDiscrete; //P constructs a classifier for imputation of discrete values
  PLearner learnerForContinuous; //P constructs a classifier for imputation of continuous values

  virtual PDomainContingency operator()(PExampleGenerator, const long &weightID=0);
};


class ORANGE_API TComputeDomainContingency_Preprocessor : public TComputeDomainContingency {
public:
  __REGISTER_CLASS

  // This preprocessor shouldn't change the domain, only select examples! (should be a PFilter, actually)
  PPreprocessor preprocessor; //P preprocesses the exmaples (see the manual for restrictions!)

  bool resultInOriginalDomain; //P tells whether the resulting DomainContingency should consist of original attributes
  // See the comment at TComputeDomainContingency_DomainTransformation for information on
  // the meaning of resultInOriginalDomain and the corresponding restrictions on the preprocessor

  virtual PDomainContingency operator()(PExampleGenerator, const long &weightID=0);
};

WRAPPER(ComputeDomainContingency_DomainTransformation)
WRAPPER(ComputeDomainContingency_Preprocessor)

#endif
