/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


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
