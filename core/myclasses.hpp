// We need this for MYMODULE_API and for REGISTER_CLASS
#include "mymodule_globals.hpp"

#include "learn.hpp"
#include "classify.hpp"
#include "random.hpp"
#include "filter.hpp"

// Class definitions as usual, except for the MYMODULE_API, __REGISTER_CLASS and //P

class CORE_API TRandomForestLearner : public TLearner {
public:
    __REGISTER_CLASS

    int randomSeed;  //P seed for the random generator

    TRandomForestLearner(const int &seed = 0);
    PClassifier operator()(PExampleGenerator, const int &weightID);
};


class CORE_API TRandomForest : public TClassifier {
public:
    __REGISTER_CLASS

    PRandomGenerator randomGenerator; //PR random generator

    TRandomForest(PVariable classVar, PRandomGenerator rgen);
    virtual TValue operator()(const TExample &ex);
};
