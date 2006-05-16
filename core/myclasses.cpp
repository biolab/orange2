// all other files should include myclasses.hpp (if they need these definitions),
// myclasses.cpp should be the only one who includes myclasses.ppp
#include "myclasses.ppp"

#include "errors.hpp" // for raiseError

TMyLearner::TMyLearner(const int &seed)
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


/* The only fancy thing in this file: sumValues is actually CRC32,
   so the result of this function is semi-random, but always the
   same for the same example */
bool TMyFilter::operator()(const TExample &ex)
{
  return ex.sumValues() & 1;
}
