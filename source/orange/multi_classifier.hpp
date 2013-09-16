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
