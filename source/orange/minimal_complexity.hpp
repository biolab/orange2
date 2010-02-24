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


#ifndef __MINIMAL_COMPLEXITY_HPP
#define __MINIMAL_COMPLEXITY_HPP

#include "induce.hpp"
#include "exampleclustering.hpp"
#include "decomposition.hpp"

WRAPPER(Example)


class ORANGE_API TIGNode {
public:
  PExample example;
  TDiscDistribution incompatibility, compatibility;
  int randint;

  TIGNode();
  TIGNode(PExample);
  TIGNode(PExample, const TDiscDistribution &, const TDiscDistribution &);
};

/*  Incompatibility graph; each element of the vector holds a combination of values of bound
    attributes and a distribution of incompatibilities with other graph nodes. Additional methods are
    provided for removing not connected nodes, making the incompatibility weights 0 or 1, and for
    normalizing them. */
class ORANGE_API TIG : public TOrange {
public:
  __REGISTER_CLASS

  vector<TIGNode> nodes;
  bool checkedForEmpty;

  TIG();

  int traverse(visitproc visit, void *arg) const;
  int dropReferences();

  void removeEmpty();
  void make0or1();
  void normalize();
  void complete();
};

WRAPPER(IG);


class ORANGE_API TIGConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual PIG operator()(PExampleGenerator, TVarList &boundSet, const int &weight) =0;
};

WRAPPER(IGConstructor);


class ORANGE_API TIGByIM : public TIGConstructor  {
public:
  __REGISTER_CLASS

  PIMConstructor IMconstructor; //P IM constructor

  virtual PIG operator()(PExampleGenerator, TVarList &boundSet, const int &weight);
};


class ORANGE_API TIGBySorting: public TIGConstructor {
public:
  __REGISTER_CLASS

  virtual PIG operator()(PExampleGenerator, TVarList &aboundSet, const int &weight);
};



class ORANGE_API TColoredIG : public TGeneralExampleClustering {
public:
  __REGISTER_CLASS

  PIG ig; //P incompatibility graph
  PIntList colors; //P colors (one element corresponding to each ig node)

  TColoredIG(PIG = PIG());

  PExampleClusters exampleClusters() const;
  PExampleSets exampleSets(const float &) const;
};

WRAPPER(ColoredIG);


class ORANGE_API TColorIG : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PColoredIG operator()(PIG)=0;
};

WRAPPER(ColorIG);


class ORANGE_API TColorIG_MCF : public TColorIG {
public:
  __REGISTER_CLASS

  virtual PColoredIG operator()(PIG);
};


class ORANGE_API TFeatureByMinComplexity : public TFeatureInducer {
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(Completion: NoCompletion=completion_no; CompletionByDefault=completion_default; CompletionByBayes=completion_bayes)

  PColorIG colorIG; //P graph coloring algorithm
  int completion; //P(&FeatureByMinComplexity_Completion) decides how to determine the class for points not covered by any cluster

  TFeatureByMinComplexity(PColorIG = PColorIG(), const int &completion = completion_bayes);
  PVariable operator()(PExampleGenerator gen, TVarList &boundSet, const string &name, float &quality, const int &weight=0);
};

#endif
