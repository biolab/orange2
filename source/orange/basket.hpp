#ifndef __BASKET_HPP
#define __BASKET_HPP

#include <map>
#include <string>

#include "filegen.hpp"
#include "domain.hpp"

using namespace std;

class ORANGE_API TBasketFeeder : public TOrange {
public:
  __REGISTER_CLASS

  bool dontStore; //P disables items storing
  bool dontCheckStored; //P disables items lookup in the global cache
  PDomain domain; //P domain where the meta attributes are stored
  PDomain sourceDomain; //P domain with items that can be reused

  TBasketFeeder(PDomain sourceDomain, bool dontCheckStored, bool dontStore);

  void addItem(TExample &example, const string &atom, const int &lineno);
  static void clearCache();

protected:
  map<string, int> localStore;

  static map<string, TMetaDescriptor> itemCache;
};

WRAPPER(BasketFeeder);


class ORANGE_API TBasketExampleGenerator : public TFileExampleGenerator {
public:
  __REGISTER_CLASS
  PBasketFeeder basketFeeder;

  TBasketExampleGenerator(const string &datafile, PDomain sourceDomain, const int createNewOn, vector<int> &status, vector<pair<int, int> > &metaStatus);

  virtual bool readExample(TFileExampleIteratorData &, TExample &);
};

#endif