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

	// statistics computation 
	// Wald Z statistic (PFloatList beta, PFloatList beta_se)
	PFloatList computeWaldZ(PFloatList &, PFloatList &);
	// P for chi square (PFloatList wald_Z)
	PFloatList computeP(PFloatList &);

	bool showSingularity; //P Defines whether singularity should be thrown as error

	// Constructs a Logistic classifier 
	// weights are not implemented at the moment
	virtual PClassifier operator()(PExampleGenerator gen, const int & = 0);
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
	// Wald Z Statistic
	PFloatList wald_Z; //P Wald Z statstic for beta coefficients
	// P
	PFloatList P; //P estimated significances for beta coefficients
	// likelihood
	float likelihood; //P Likelihood: The likelihood function is the function which specifies the probability of the sample observed on the basis of a known model, as a function of the model's parameters. 
	// error
	int error; //P Error code thrown by the selected fitter. 0(zero) means that no errors occured while fitting.
	PVariable error_att; //P Attribute that causes singularity if it occurs. 

	// constructors
	TLogisticClassifier();
	TLogisticClassifier(PDomain);

	virtual PDistribution classDistribution(const TExample &ex);
/*	{ TExample example(domain, ex);
	   example * beta */
};


#endif










