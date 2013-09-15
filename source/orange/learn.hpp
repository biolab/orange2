#ifndef __LEARN_HPP
#define __LEARN_HPP


WRAPPER(Variable)
WRAPPER(Distribution)
WRAPPER(DomainDistributions)
WRAPPER(DomainContingency)
WRAPPER(ExampleGenerator)

/*  A base for classes which can learn to classify examples after presented an appropriate learning set.
    Learning is invoked by calling 'learn' method and can be forgoten by calling 'forget'. */
class ORANGE_API TLearner : public TOrange {
public:
  __REGISTER_CLASS

  enum {NeedsNothing, NeedsClassDistribution, NeedsDomainDistribution, NeedsDomainContingency, NeedsExampleGenerator};
  int needs; //PR the kind of data that learner needs

  TLearner(const int & = NeedsExampleGenerator);
  
  virtual PClassifier operator()(PVariable);
  virtual PClassifier operator()(PDistribution);
  virtual PClassifier operator()(PDomainDistributions);
  virtual PClassifier operator()(PDomainContingency);
  virtual PClassifier operator()(PExampleGenerator, const int &weight = 0);

  virtual PClassifier smartLearn(PExampleGenerator, const int &weight,
	                               PDomainContingency = PDomainContingency(),
                                 PDomainDistributions = PDomainDistributions(),
                                 PDistribution = PDistribution());
};


class ORANGE_API TLearnerFD : public TLearner {
public:
  __REGISTER_CLASS

  PDomain domain; //P domain

  TLearnerFD();
  TLearnerFD(PDomain);
};

WRAPPER(Learner)

#endif

