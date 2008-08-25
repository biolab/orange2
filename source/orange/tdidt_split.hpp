/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifndef __TDIDT_SPLIT_HPP
#define __TDIDT_SPLIT_HPP

#include "root.hpp"
#include "transval.hpp"
#include "orvector.hpp"

#include "examplegen.hpp"
#include <limits>

WRAPPER(Classifier)
WRAPPER(DiscDistribution)
WRAPPER(DomainContingency)
WRAPPER(ExampleGenerator)
WRAPPER(Distribution)
WRAPPER(MeasureAttribute)
WRAPPER(ExampleTable)


class ORANGE_API TTreeSplitConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  float minSubset; //P minimal number of examples in a subset

  TTreeSplitConstructor(const float &aml = 0);

  /*  Abstract method which returns a split criteria for the given set of examples.
      IF domainContingency is given the method may use it instead of the example generator.
      Split criteria is returned in form of TSelectBranch, which returns an index of subset for an example.
      If no criteria is found, NULL can be returned to stop the construction. */
  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               )=0;

  inline PClassifier returnNothing(PStringList &description, PDiscDistribution &subsetSizes, int &spentAttribute)
  { description = PStringList();
    subsetSizes = PDiscDistribution();
    spentAttribute = -1;
    return PClassifier();
  }

  inline PClassifier returnNothing(PStringList &description, PDiscDistribution &subsetSizes, float &quality, int &spentAttribute)
  { description = PStringList();
    subsetSizes = PDiscDistribution();
    quality = numeric_limits<float>::quiet_NaN();
    spentAttribute = -1;
    return PClassifier();
  }
};

WRAPPER(TreeSplitConstructor);


class ORANGE_API TTreeSplitConstructor_Measure : public TTreeSplitConstructor {
public:
  __REGISTER_ABSTRACT_CLASS

  PMeasureAttribute measure; //P attribute quality measure
  float worstAcceptable; //P the worst acceptable quality of the attribute

  TTreeSplitConstructor_Measure(PMeasureAttribute = PMeasureAttribute(), const float &worst = 0, const float &aminSubset = 0.0);
};


class ORANGE_API TTreeSplitConstructor_Combined : public TTreeSplitConstructor {
public:
  __REGISTER_CLASS

  PTreeSplitConstructor discreteSplitConstructor; //P split constructor for discrete attributes
  PTreeSplitConstructor continuousSplitConstructor; //P split constructor for continuous attributes

  TTreeSplitConstructor_Combined(PTreeSplitConstructor = PTreeSplitConstructor(), PTreeSplitConstructor = PTreeSplitConstructor(), const float & = 0);

  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               );
};


class ORANGE_API TTreeSplitConstructor_Attribute : public TTreeSplitConstructor_Measure {
public:
  __REGISTER_CLASS

  TTreeSplitConstructor_Attribute(PMeasureAttribute = PMeasureAttribute(), const float &worst = -1e30, const float & = 0.0);

  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               );
};


class ORANGE_API TTreeSplitConstructor_ExhaustiveBinary : public TTreeSplitConstructor_Measure {
public:
  __REGISTER_CLASS

  TTreeSplitConstructor_ExhaustiveBinary(PMeasureAttribute = PMeasureAttribute(), const float &worst = -1e30, const float & = 0.0);

  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               );
};

class ORANGE_API TTreeSplitConstructor_OneAgainstOthers: public TTreeSplitConstructor_Measure {
public:
  __REGISTER_CLASS
  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               );
};


class ORANGE_API TTreeSplitConstructor_Threshold: public TTreeSplitConstructor_Measure {
public:
  __REGISTER_CLASS

  TTreeSplitConstructor_Threshold(PMeasureAttribute = PMeasureAttribute(), const float &worst = -1e30, const float & = 0);

  virtual PClassifier operator()(PStringList &descriptions,
                                PDiscDistribution &subsetSizes,
                                float &quality, int &spentAttribute,

                                PExampleGenerator, const int &weightID = 0,
                                PDomainContingency = PDomainContingency(),
                                PDistribution apriorClass = PDistribution(),
                                const vector<bool> &candidates = vector<bool>(),
                                PClassifier nodeClassifier = PClassifier()
                               );
};



/* The following classes assumge the the given ExampleGenerator has
   fixed examples and return ExampleTable with references.
   If the splitter returns non-zero newWeight, weight meta-attribute
   must be removed by the caller (when not needed any more).

   There are two methods -- single-pass split and one-by-one split.
   'singlePass' determines which of the two is implemented.
   In single-pass split, operator returns a vector of TExampleTables.
   In one-by-one split, caller should specify a branch index and
   TExampleTable with the corresponding examples are returned.
*/
WRAPPER(TreeNode)


class ORANGE_API TTreeExampleSplitter : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights) =0;

protected:
  static PExampleGeneratorList prepareGeneratorList(int size, PExampleGenerator generator, vector<TExampleTable *> &);
  static bool getBranchIndices(PTreeNode node, PExampleGenerator generator, vector<int> &indices);
};


class ORANGE_API TTreeExampleSplitter_IgnoreUnknowns : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


class ORANGE_API TTreeExampleSplitter_UnknownsToCommon : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


class ORANGE_API TTreeExampleSplitter_UnknownsToAll : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


class ORANGE_API TTreeExampleSplitter_UnknownsToRandom : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


class ORANGE_API TTreeExampleSplitter_UnknownsToBranch : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};


class ORANGE_API TTreeExampleSplitter_UnknownsAsBranchSizes : public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};

class ORANGE_API TTreeExampleSplitter_UnknownsAsSelector: public TTreeExampleSplitter {
public:
  __REGISTER_CLASS
  virtual PExampleGeneratorList operator()(PTreeNode node, PExampleGenerator generator, const int &weightID, vector<int> &weights);
};

#endif
