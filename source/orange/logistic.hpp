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

    Authors: Martin Mozina, Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/

#ifdef _MSC_VER
  #pragma warning (disable : 4786)
#endif

#ifndef __LOGISTIC_HPP
#define __LOGISTIC_HPP


// include
#include "classify.hpp"
#include "learn.hpp"
#include "logfit.hpp"

// TODO: add other includings


// Logistic regression learner
class TLogisticLearner : public TLearner {
public:
	__REGISTER_CLASS

	// fitter
	PLogisticFitter fitter;

	// constructors
	TLogisticLearner();

	// Constructs a Logistic classifier
	virtual PClassifier operator()(PExampleGenerator gen, const int & =0);

};



// Logistic regression classifier
// coefficients are needed for each attribute
class TLogisticClassifier : public TClassifierFD {
public:
	__REGISTER_CLASS

	// coeficients
	PFloatList beta; //P estimated beta coefficients for logistic regression
	// beta standard errors
	PFloatList beta_se; //P estimated standard errors for beta coefficients
	
	// constructors
	TLogisticClassifier();
	TLogisticClassifier(PDomain);

	virtual PDistribution classDistribution(const TExample &ex);
/*	{ TExample example(domain, ex);
	   example * beta */
};


#endif










