#ifndef __REDUNDANCY_HPP
#define __REDUNDANCY_HPP

WRAPPER(Discretization)
WRAPPER(MeasureAttribute)
WRAPPER(FeatureInducer)

class ORANGE_API TRemoveRedundant : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool keepValues; //P keep an attribute if number values is only smaller, not one

  TRemoveRedundant(bool akeep=false);
  virtual PDomain operator()(PExampleGenerator gen, PVarList suspicious=PVarList(), PExampleGenerator *nonRedundantResult=NULL, int aweightID=0)=0;
};


class ORANGE_API TRemoveRedundantByInduction : public TRemoveRedundant {
public:
  __REGISTER_CLASS

  PFeatureInducer featureReducer; //P feature construction algorithm
  PMeasureAttribute measure; //P measure for initial ordering of attributes
  PDiscretization discretization; //P discretization method

  TRemoveRedundantByInduction(bool akeepValues=false);
  virtual PDomain operator()(PExampleGenerator gen, PVarList suspicious=PVarList(), PExampleGenerator *nonRedundantResult=NULL, int aweightID=0);
};


class ORANGE_API TRemoveRedundantByQuality : public TRemoveRedundant {
public:
  __REGISTER_CLASS

  bool remeasure; //P reapply the measure after removal
  float minQuality; //P minimal acceptable quality
  int removeBut; //P the desired number of attributes
  PMeasureAttribute measure; //P attribute quality measure

  TRemoveRedundantByQuality(bool aremeasure=false);
  virtual PDomain operator()(PExampleGenerator gen, PVarList suspicious=PVarList(), PExampleGenerator *nonRedundantResult=NULL, int aweightID=0);
};


class ORANGE_API TRemoveRedundantOneValue : public TRemoveRedundant {
public:
  __REGISTER_CLASS

  bool onData; //P observe the actual number of value on the data (not on attribute definitions)

  TRemoveRedundantOneValue(bool anOnData=true);
  virtual PDomain operator()(PExampleGenerator gen, PVarList suspicious=PVarList(), PExampleGenerator *nonRedundantResult=NULL, int aweightID=0);
  static bool hasAtLeastTwo(PExampleGenerator gen, const int &idx);
};

/*
class ORANGE_API TRemoveNonexistentValues : public TRemoveRedundant {
public:
  _ _ R E G I S T E R _ C L A S S 

  virtual PDomain operator()(PExampleGenerator gen, PVarList suspicious=PVarList(), PExampleGenerator *nonRedundantResult=NULL, int aweightID=0);
};
*/

class ORANGE_API TRemoveUnusedValues : public TOrange {
public:
  __REGISTER_CLASS

  TRemoveUnusedValues(bool = false);

  bool removeOneValued; //P if true (default is false), one valued attributes are also removed
  virtual PVariable operator()(PVariable, PExampleGenerator, const int &);
  PDomain operator ()(PExampleGenerator gen, const int &weightID, bool checkClass = false, bool checkMetas = true);
};

#endif
