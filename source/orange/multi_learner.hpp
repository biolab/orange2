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

#ifndef __MULTI_LEARNER_HPP
#define __MULTI_LEARNER_HPP

#include "root.hpp"
#include "multi_classifier.hpp"

WRAPPER (Variable)
WRAPPER (Distribution)
WRAPPER (DomainDistributions)
WRAPPER (DomainContingency)
WRAPPER (ExampleGenerator)

/*  A base for classes which can learn to classify examples after presented an appropriate learning set.
 Learning is invoked by calling 'learn' method and can be forgoten by calling 'forget'. */
class ORANGE_API TMultiLearner : public TOrange {
public:
	__REGISTER_CLASS

	enum {NeedsNothing, NeedsClassDistribution, NeedsDomainDistribution, NeedsDomainContingency, NeedsExampleGenerator};
	int needs; //PR the kind of data that learner needs

	PDomain domain; //P domain

	TMultiLearner(const int & = NeedsExampleGenerator);

	virtual PMultiClassifier operator()(PVarList);
	virtual PMultiClassifier operator()(PDistributionList);
	virtual PMultiClassifier operator()(PDomainDistributions);
	virtual PMultiClassifier operator()(PDomainContingency);
	virtual PMultiClassifier operator()(PExampleGenerator, const int &weight = 0);

};

WRAPPER(MultiLearner)

#endif

