#ifndef __CARTESIAN_HPP
#define __CARTESIAN_HPP

#include "classify.hpp"
#include "values.hpp"
#include "examples.hpp"

class ORANGE_API TCartesianClassifier : public TClassifierFD {
public:
  __REGISTER_CLASS

  virtual TValue operator ()(const TExample &);

  virtual void afterSet(const char *name);
  void domainHasChanged();

protected:
  vector<int> mults;
};

#endif
