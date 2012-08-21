/*
 This file is part of Orange.

 Copyright 1996-2012 Faculty of Computer and Information Science, University of Ljubljana
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

#ifndef _MULTI_CLASSIFIER_HPP
#define _MULTI_CLASSIFIER_HPP

#include <string>
#include "root.hpp"
#include "distvars.hpp"

using namespace std;
WRAPPER(MultiClassifier)
WRAPPER(EFMDataDescription);

#define TMultiClassifierList TOrangeVector<PMultiClassifier> 
VWRAPPER(MultiClassifierList)

/* Classifiers have three methods for classification.
 - operator() returns TValue
 - classDistribution return PDistribution
 - predictionAndDistribution returns both

 At least one of the first two need to be overloaded. If the method
 can return probabilities (or at least something the closely
 ressembles it), it should redefine the second.
 */

class ORANGE_API TMultiClassifier: public TOrange {
public:
	__REGISTER_CLASS

	PVarList classVars; //P class variables
	PDomain domain; //P domain
	bool computesProbabilities; //P set if classifier computes class probabilities (if not, it assigns 1.0 to the predicted)

	TMultiClassifier(const bool &cp = false);
	TMultiClassifier(const PVarList &, const bool &cp = false);
	TMultiClassifier(const TMultiClassifier &old);

	virtual PValueList operator ()(const TExample &);
	virtual PDistributionList classDistribution(const TExample &);
	virtual void predictionAndDistribution(const TExample &, PValueList &,
			PDistributionList &);
};


#endif
