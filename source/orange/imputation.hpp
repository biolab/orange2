#ifndef __IMPUTATION_HPP
#define __IMPUTATION_HPP

#include "root.hpp"

WRAPPER(Imputer)
WRAPPER(ExampleGenerator)
WRAPPER(ImputerConstructor)
WRAPPER(Example)
WRAPPER(Learner)
VWRAPPER(ClassifierList)


class TImputer : public TOrange
{
public:
  __REGISTER_ABSTRACT_CLASS
  virtual TExample *operator()(TExample &) = 0;

  virtual PExampleGenerator operator()(PExampleGenerator, const int &);
};


class TImputer_defaults : public TImputer
{
public:
  __REGISTER_CLASS
  PExample defaults; //P values that are to be inserted instead of missing ones

  TImputer_defaults(PDomain domain);
  TImputer_defaults(const TExample &example);
  virtual TExample *operator()(TExample &);
};


class TImputer_asValue : public TImputer
{
public:
  __REGISTER_CLASS
  PDomain domain; //P domain to which the values are converted
  virtual TExample *operator()(TExample &example);
};


class TImputer_model : public TImputer
{
public:
  __REGISTER_CLASS

  PClassifierList models;
  virtual TExample *operator()(TExample &example);
};


class TImputerConstructor : public TOrange
{
public:
  __REGISTER_ABSTRACT_CLASS

  bool imputeClass; //P tells whether to impute the class value (default: true)

  TImputerConstructor();
  virtual PImputer operator()(PExampleGenerator, const int &) = 0;
};


class TImputerConstructor_defaults : public TImputerConstructor
{
public:
  __REGISTER_CLASS
  PExample defaults; //P default values to be imputed instead missing ones

  virtual PImputer operator()(PExampleGenerator, const int &);
};

class TImputerConstructor_average : public TImputerConstructor
{
public:
  __REGISTER_CLASS
  virtual PImputer operator()(PExampleGenerator, const int &);
};


class TImputerConstructor_minimal : public TImputerConstructor
{
public:
  __REGISTER_CLASS
  virtual PImputer operator()(PExampleGenerator, const int &);
};


class TImputerConstructor_maximal : public TImputerConstructor
{
public:
  __REGISTER_CLASS
  virtual PImputer operator()(PExampleGenerator, const int &);
};


WRAPPER(Learner)
class TImputerConstructor_model : public TImputerConstructor
{
public:
  __REGISTER_CLASS
  PLearner learnerDiscrete; //P learner for discrete attributes
  PLearner learnerContinuous; //P learner for continuous attributes

  bool useClass; //P tells whether to use class value in imputation (default: false)

  TImputerConstructor_model();
  virtual PImputer operator()(PExampleGenerator, const int &);
};

#endif