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

#include "vars.hpp"
#include <math.h>
#include "stladdon.hpp"
#include "random.hpp"
#include "examplegen.hpp"
#include "examples.hpp"
#include "domain.hpp"
//#include "filter.hpp"
#include "table.hpp"

#include "multi_classifier.ppp"

DEFINE_TOrangeVector_classDescription(PMultiClassifier, "TMultiClassifierList", true, ORANGE_API)

/* ***** TMultiClassifier methods */

TMultiClassifier::TMultiClassifier(const PVarList &acv, const bool &cp) :
		classVars(acv), computesProbabilities(cp) {
}
;

TMultiClassifier::TMultiClassifier(const bool &cp) :
		classVars(PVarList()), computesProbabilities(cp) {
}
;

TMultiClassifier::TMultiClassifier(const TMultiClassifier &old) :
		TOrange(old), classVars(old.classVars), computesProbabilities(
				old.computesProbabilities) {
}
;

PValueList TMultiClassifier::operator ()(const TExample &exam) {
	if (!computesProbabilities)
		raiseError("invalid setting of 'computesProbabilities'");

	PValueList classValues = new TValueList();
	TValue value;
	PVariable classVar;
	PDistributionList classDists = classDistribution(exam);

	for (int i = 0; i < classVars->size(); i++) {
		classVar = classVars->at(i);
		value = classVar->varType == TValue::FLOATVAR ?
				TValue(classDists->at(i)->average()) : classDists->at(i)->highestProbValue(exam);
		classValues->push_back(value);
	}

	return classValues;
}

PDistributionList TMultiClassifier::classDistribution(const TExample &exam) {
	if (computesProbabilities)
		raiseError("invalid setting of 'computesProbabilities'");

	PDistributionList classDists = new TDistributionList();
	PDistribution dist;
	PVariable classVar;
	PValueList classValues = operator()(exam);

	for (int i = 0; i < classVars->size(); i++) {
		classVar = classVars->at(i);
		dist = TDistribution::create(classVar);
		dist->add(classValues->at(i));
		classDists->push_back(dist);
	}
	return dist;
}

void TMultiClassifier::predictionAndDistribution(const TExample &ex,
		PValueList &classValues, PDistributionList &classDists) {
	if (computesProbabilities) {
		classDists = classDistribution(ex);
		PValueList classValues = new TValueList();
		TValue value;
		PVariable classVar;
		for (int i = 0; i < classVars->size(); i++) {
			classVar = classVars->at(i);
			value = classVar->varType == TValue::FLOATVAR ?
					TValue(classDists->at(i)->average()) :
					classDists->at(i)->highestProbValue(ex);
			classValues->push_back(value);
		}
	} else {
		classValues = operator()(ex);
		PDistributionList classDist = new TDistributionList();
		PDistribution dist;
		PVariable classVar;

		for (int i = 0; i < classVars->size(); i++) {
			classVar = classVars->at(i);
			dist = TDistribution::create(classVar);
			dist->add(classValues->at(i));
			classDist->push_back(dist);
		}

	}
}

