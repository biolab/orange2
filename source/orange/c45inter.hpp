#ifndef __C45INTER_HPP
#define __C45INTER_HPP

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;

class ORANGE_API TC45ExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS

  PBoolList skip;  //P a boolean list, one element per attribute, denoting which attributes to skip

  TC45ExampleGenerator(const string &datafile, const string &domainFile, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);
  TC45ExampleGenerator(const TC45ExampleGenerator &old);

  virtual bool readExample(TFileExampleIteratorData &, TExample &);

  PDomain readDomain(const string &stem, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);
};

#endif

