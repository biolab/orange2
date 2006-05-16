// We need this for MYMODULE_API and for REGISTER_CLASS
#include "mymodule_globals.hpp"

#include "learn.hpp"
#include "classify.hpp"
#include "random.hpp"
#include "filter.hpp"

// Class definitions as usual, except for the MYMODULE_API, __REGISTER_CLASS and //P

class MYMODULE_API TMyLearner : public TLearner {
public:
    __REGISTER_CLASS

    int randomSeed;  //P seed for the random generator

    TMyLearner(const int &seed = 0);
    PClassifier operator()(PExampleGenerator, const int &weightID);
};


class MYMODULE_API TMyClassifier : public TClassifier {
public:
    __REGISTER_CLASS

    PRandomGenerator randomGenerator; //PR random generator

    TMyClassifier(PVariable classVar, PRandomGenerator rgen);
    virtual TValue operator()(const TExample &ex);
};


class MYMODULE_API TMyFilter : public TFilter {
public:
    __REGISTER_CLASS

    virtual bool operator()(const TExample &);
};
