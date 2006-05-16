#include "myclasses.ppp"

#include "errors.hpp" // for raiseError

/*
TRandomForestLearner::TRandomForestLearner(const int &seed)
: randomSeed(seed)
{}


PClassifier TMyLearner::operator()(PExampleGenerator egen, const int &weightID)
{
  PRandomGenerator randGen = new TRandomGenerator(randomSeed);
  return new TMyClassifier(egen->domain->classVar, randGen);
}



TMyClassifier::TMyClassifier(PVariable classVar, PRandomGenerator rgen)
: TClassifier(classVar),
  randomGenerator(rgen)
{
  if (classVar->varType != TValue::INTVAR)
    raiseError("MyClassifier cannot work with a non-discrete attribute '%s'", classVar->name.c_str());
}


TValue TMyClassifier::operator()(const TExample &)
{
  return TValue(randomGenerator->randint(classVar->noOfValues()));
}

*/