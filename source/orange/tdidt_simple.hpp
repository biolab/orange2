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


#ifndef __TDIDT_SIMPLE_HPP
#define __TDIDT_SIMPLE_HPP

#include <limits.h>
#include "learn.hpp"

struct SimpleTreeNode {
    int type, children_size, split_attr;
    float split;
    SimpleTreeNode **children;

    float *dist;  /* classification */
    float n, sum; /* regression */
};

class ORANGE_API TSimpleTreeLearner : public TLearner {
public:
	__REGISTER_CLASS
    float maxMajority; //P
    int minInstances; //P
    int maxDepth; //P
    float skipProb; //P

	TSimpleTreeLearner(const int & =0, float=1.0, int=0, int=INT_MAX, float=0.0);
	PClassifier operator()(PExampleGenerator, const int & =0);
};

class ORANGE_API TSimpleTreeClassifier : public TClassifier {
private:
    int type;
	struct SimpleTreeNode *tree;

public:
	__REGISTER_CLASS 

	TSimpleTreeClassifier();
	TSimpleTreeClassifier(const PVariable &, struct SimpleTreeNode *tree, int type);
    ~TSimpleTreeClassifier();

	TValue operator()(const TExample &);
	PDistribution classDistribution(const TExample &);
	void predictionAndDistribution(const TExample &, TValue &, PDistribution &);
};

#endif
