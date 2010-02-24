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


#ifndef __TRINDEX_HPP
#define __TRINDEX_HPP

#include <vector>
#include "orvector.hpp"
#include "random.hpp"
using namespace std;

WRAPPER(ExampleGenerator);


// For compatibility ...
#define TFoldIndices TLongList
#define PFoldIndices PLongList
#define PRandomIndices PLongList

class ORANGE_API TMakeRandomIndices : public TOrange {
public:
  __REGISTER_CLASS
  CLASSCONSTANTS(Stratification: StratifiedIfPossible=-1; NotStratified; Stratified)

  enum { STRATIFIED_IF_POSSIBLE=-1, NOT_STRATIFIED, STRATIFIED};

  int stratified;  //P(&MakeRandomIndices_Stratification) requests stratified distributions
  int randseed; //P a seed for random generator
  PRandomGenerator randomGenerator; //P a random generator

  TMakeRandomIndices(const int &stratified=STRATIFIED_IF_POSSIBLE, const int &randseed=-1);
  TMakeRandomIndices(const int &stratified, PRandomGenerator);
};

WRAPPER(MakeRandomIndices)

// Prepares a vector of 0's and 1's with given distribution.
class ORANGE_API TMakeRandomIndices2 : public TMakeRandomIndices {
public:
  __REGISTER_CLASS

  float p0; //P a proportion or a number of 0's

  TMakeRandomIndices2(const float &p0=1.0, const int &stratified=TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, const int &randseed=-1);
  TMakeRandomIndices2(const float &p0, const int &stratified, PRandomGenerator);

  PRandomIndices operator()(const int &n);
  PRandomIndices operator()(const int &n, const float &p0);

  PRandomIndices operator()(PExampleGenerator);
  PRandomIndices operator()(PExampleGenerator, const float &p0);
};



/*  Prepares a vector of indices with given distribution. Similar to TMakeRandomIndices2, this object's constructor
    is given the size of vector and the required distribution of indices. */
class ORANGE_API TMakeRandomIndicesN : public TMakeRandomIndices {
public:
  __REGISTER_CLASS

  PFloatList p; //P probabilities of indices (last is 1-sum(p))

  TMakeRandomIndicesN(const int &stratified=TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, const int &randseed=-1);
  TMakeRandomIndicesN(PFloatList p, const int &stratified=TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, const int &randseed=-1);

  TMakeRandomIndicesN(const int &stratified, PRandomGenerator);
  TMakeRandomIndicesN(PFloatList p, const int &stratified, PRandomGenerator);

  PRandomIndices operator()(const int &n);
  PRandomIndices operator()(const int &n, PFloatList p);

  PRandomIndices operator()(PExampleGenerator);
  PRandomIndices operator()(PExampleGenerator, PFloatList p);
};


// Indices for cross validation
class ORANGE_API TMakeRandomIndicesCV : public TMakeRandomIndices {
public:
  __REGISTER_CLASS

  int folds; //P number of folds

  TMakeRandomIndicesCV(const int &folds=10, const int &stratified=TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, const int &randseed=-1);
  TMakeRandomIndicesCV(const int &folds, const int &stratified, PRandomGenerator);

  PRandomIndices operator()(const int &n);
  PRandomIndices operator()(const int &n, const int &folds);

  PRandomIndices operator()(PExampleGenerator);
  PRandomIndices operator()(PExampleGenerator, const int &folds);
};


// Prepares a vector of 0's and 1's with given distribution.
class ORANGE_API TMakeRandomIndicesMultiple : public TMakeRandomIndices {
public:
  __REGISTER_CLASS

  float p0; // proportion/number of examples

  TMakeRandomIndicesMultiple(const float &p0=1.0, const int &stratified=TMakeRandomIndices::STRATIFIED_IF_POSSIBLE, const int &randseed=-1);
  TMakeRandomIndicesMultiple(const float &p0, const int &stratified, PRandomGenerator);

  PRandomIndices operator()(const int &n);
  PRandomIndices operator()(const int &n, const float &p0);

  PRandomIndices operator()(PExampleGenerator);
  PRandomIndices operator()(PExampleGenerator, const float &p0);

};


#endif

