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

