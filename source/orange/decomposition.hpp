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


#ifndef __DECOMPOSITION_HPP
#define __DECOMPOSITION_HPP

#include "distvars.hpp"

#include <vector>


WRAPPER(Example)

class TExample_nodeIndex
{ public:
    PExample example;
    int nodeIndex;

    TExample_nodeIndex(PExample);
};

typedef vector<TExample_nodeIndex>::iterator TEnIIterator;

class TSortedExamples_nodeIndices : public vector<TExample_nodeIndex>
{ public: 
    PExampleGenerator exampleTable;
    int maxIndex;

    TSortedExamples_nodeIndices(PExampleGenerator eg, const vector<bool> &bound, const vector<bool> &free);
    void sortByAttr(int attrNo, vector<TEnIIterator> *&sorting, int values);
    void sortByAttr_Mult(int attrNo, vector<TEnIIterator> *&sorting, int values);
};



/* Incompatibility matrix */

class TIMColumnNode {
public:
  int index;
  TIMColumnNode *next;
  float nodeQuality;    // not necessarily defined!

  TIMColumnNode(const int &ind, TIMColumnNode * =NULL, float nerr=0.0);
  virtual ~TIMColumnNode();

  virtual TIMColumnNode &operator += (const TIMColumnNode &)=0;
};


class TDIMColumnNode : public TIMColumnNode {
public:
  int noOfValues;
  float *distribution;
  float abs;

  TDIMColumnNode(const int &ind, const int &noOfValues, float * =NULL, TIMColumnNode * =NULL);
  virtual ~TDIMColumnNode();

  virtual TIMColumnNode &operator += (const TIMColumnNode &);
  inline void computeabs()
  { abs = 0.0;
    float *di = distribution;
    for(int c = noOfValues; c--; abs += *(di++));
  }
};


class TFIMColumnNode : public TIMColumnNode {
public:
  float sum, sum2, N;

  TFIMColumnNode(int, TIMColumnNode * =NULL, const float asum=0.0, const float asum2=0.0, const float aN=0.0);
  void add(const float &value, const float weight=1.0);

  virtual TIMColumnNode &operator += (const TIMColumnNode &);
};


class T_ExampleIMColumnNode {
public:
  PExample example;
  TIMColumnNode *column;

  T_ExampleIMColumnNode(PExample anexample=PExample(), TIMColumnNode *anode=NULL);
  T_ExampleIMColumnNode(const T_ExampleIMColumnNode &other);
  ~T_ExampleIMColumnNode();

  T_ExampleIMColumnNode &operator =(const T_ExampleIMColumnNode &other);
};


WRAPPER(ExampleTable)

class TIM : public TOrange {
public:
  __REGISTER_CLASS

  int varType; //P class variable type
  PExampleTable rowExamples; //P examples with free attributes for each row
  vector<T_ExampleIMColumnNode> columns;

  TIM(const int &aVarType);
  int traverse(visitproc visit, void *arg) const;
  int dropReferences();
  bool fuzzy();
};

WRAPPER(IM);
WRAPPER(IMByRows);


/* An abstract class to construct the incompatibility matrix. */
class TIMConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  bool recordRowExamples; //P enables creation of rowExample list

  TIMConstructor(const bool &anRE=false);

  virtual PIM operator()(PExampleGenerator, const TVarList &boundSet, const int &weightID=0);
  virtual PIM operator()(PExampleGenerator, const TVarList &boundSet, const TVarList &freeSet, const int &weightID=0);
  virtual PIM operator ()(PIMByRows imrows);

  virtual PIM operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0) =0;
};

WRAPPER(IMConstructor);


class TIMBySorting : public TIMConstructor {
public:
  __REGISTER_CLASS
  virtual PIM operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};


/* A class to obtain the incompatibility matrix from examples; not defined yet. */
class TIMFromExamples : public TOrange {
  __REGISTER_ABSTRACT_CLASS
  virtual PIM operator()(PExampleGenerator, TVarList &, const int &weightID=0) =0;
};


class TPreprocessIM : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual bool operator()(PIM)=0;
};



class TDIMRow {
public:
  PExample example;
  int noOfValues;
  vector<float *> nodes;

  TDIMRow(PExample, const int &n, const int &classes);
  virtual ~TDIMRow();
};


class TIMByRows : public TOrange {
public:
  __REGISTER_CLASS

  int varType; //P class variable type
  vector<PExample> columnExamples;
  vector<TDIMRow> rows;

  TIMByRows(const int &avarType);
  int traverse(visitproc visit, void *arg) const;
  int dropReferences();
};

WRAPPER(IMByRows)


class TIMByRowsConstructor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS

  virtual PIMByRows operator()(PExampleGenerator, const TVarList &boundSet, const int &weightID=0);
  virtual PIMByRows operator()(PExampleGenerator, const TVarList &boundSet, const TVarList &freeSet, const int &weightID=0);
  virtual PIMByRows operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0)=0;
};  


WRAPPER(IMByRowsConstructor)


class TIMByRowsBySorting : public TIMByRowsConstructor {
public:
  __REGISTER_CLASS
  virtual PIMByRows operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};  


WRAPPER(ExamplesDistance_Relief)

class TIMByRowsByRelief : public TIMByRowsConstructor {
public:
  __REGISTER_CLASS

  float k; //P number of neighbours
  float m; //P number of reference examples
  float kFromColumns; //P if positive, number of neighbours is #columns*kFromColumns

  bool ignoreSameExample; //P does not put reference example into M
  bool convertToBinary; //P convert to binary class (hit-miss)
  bool correctClassFirst; //P puts the correct class proportion to the first place
  bool allExamples; //P uses all examples for reference examples
  bool allSameNeighbours; //P uses all the examples same to the reference as neighbours

  TIMByRowsByRelief();
  virtual PIMByRows operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};  



class TIMByIMByRows : public TIMConstructor {
public:
  __REGISTER_CLASS
  virtual PIM operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};


class TIMByRelief: public TIMConstructor {
public:
  __REGISTER_CLASS

  PExamplesDistance_Relief distance; //P distance measure
  float k; //P number of neighbours
  float m; //P number of reference examples
  float kFromColumns; //P if positive, number of neighbours is #columns*kFromColumns

  bool ignoreSameExample; //P does not put reference example into M
  bool convertToBinary; //P convert to binary class (hit-miss)
  bool correctClassFirst; //P puts the correct class proportion to the first place
  bool allExamples; //P uses all examples for reference examples
  bool allSameNeighbours; //P uses all the examples same to the reference as neighbours

  TIMByRelief();
  virtual PIM operator()(PExampleGenerator, const vector<bool> &bound, const TVarList &boundSet, const vector<bool> &free, const int &weightID=0);
};



class TIMByRowsPreprocessor : public TOrange {
public:
  __REGISTER_ABSTRACT_CLASS
  virtual bool operator()(PIMByRows)=0;
};



class TIMBlurer : public TIMByRowsPreprocessor {
public:
  __REGISTER_CLASS

  float weight; //P weight of neighbours
  float origWeight; //P weight of original row
  PFloatList attrWeights; //P weights by individual (different) attributes
  bool adjustOrigWeight; //P uses 1-weight(s) for weight of original row
  bool onlyEmpty; //P blurs only empty cells

  TIMBlurer(const float &weight=1.0, const float &origWeight=1.0, const bool &aow=false, const bool &oe=false);
  TIMBlurer(PFloatList aaweights, const float &origWeight=1.0, const bool &aow=false, const bool &oe=false);

  virtual bool operator()(PIMByRows);
};

#endif
