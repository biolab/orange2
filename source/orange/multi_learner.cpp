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

// to include Python.h before STL defines a template set (doesn't work with VC >6.0)
#include "garbage.hpp" 

#include "domain.hpp"
#include "distvars.hpp"
#include "contingency.hpp"
#include "examplegen.hpp"

#include "multi_learner.ppp"


TMultiLearner::TMultiLearner(const int &aneeds) :
		needs(aneeds) {
}

PMultiClassifier TMultiLearner::operator()(PVarList) {
	if (needs == NeedsNothing)
		raiseError("invalid value of 'needs'");
	else
		raiseError("no examples");
	return PMultiClassifier();
}

PMultiClassifier TMultiLearner::operator()(PDistributionList dist) {
	switch (needs) {
	case NeedsNothing: {
		PVarList classVars = new TVarList();
		for (int i = 0; i < dist->size(); i++)
			classVars->push_back(dist->at(i)->variable);
		return operator()(classVars);
	}
	case NeedsClassDistribution:
		raiseError("invalid value of 'needs'");
		break;
	default:
		raiseError("cannot learn from class distribution only");
		break;
	};
	return PMultiClassifier();
}

PMultiClassifier TMultiLearner::operator()(PDomainDistributions ddist) {
	raiseError("NOT IMPLEMENTED"); 
	return PMultiClassifier();
}

PMultiClassifier TMultiLearner::operator()(PDomainContingency dcont) {
	raiseError("NOT IMPLEMENTED");
	return PMultiClassifier();
}

PMultiClassifier TMultiLearner::operator()(PExampleGenerator gen, const int &weight) {
	if (!gen || !gen->domain)
		raiseError("TMultiLearner: no examples or invalid example generator");
	if (!gen->domain->classVars)
		raiseError("class-less domain");

	switch (needs) {
	case NeedsNothing:
		return operator()(gen->domain->classVars);
	default:
		raiseError("invalid value of 'needs'");
		break;
	}
	return PMultiClassifier();
}

