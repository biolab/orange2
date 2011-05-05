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

#ifdef _MSC_VER
  #pragma warning (disable : 4786)
#endif

#ifndef __LOGISTIC_HPP
#define __LOGISTIC_HPP


// include
#include "classify.hpp"
#include "learn.hpp"
#include "logfit.hpp"
#include "imputation.hpp"
#include "transval.hpp"

// TODO: add other includings


// Logistic regression learner
class ORANGE_API TLogRegLearner : public TLearner {
public:
	__REGISTER_CLASS

	// fitter
	PLogRegFitter fitter; //P fits beta coefficients and calculates beta errors.

    PImputerConstructor imputerConstructor; //P if present, it constructs an imputer for unknown values
    PDomainContinuizer domainContinuizer; //P if present, it constructs continuous domain if needed; if absent, default is used

	// constructors
	TLogRegLearner();

	// statistics computation 
	// Wald Z statistic (PFloatList beta, PFloatList beta_se)
	PAttributedFloatList computeWaldZ(PAttributedFloatList &, PAttributedFloatList &);
	// P for chi square (PFloatList wald_Z)
	PAttributedFloatList computeP(PAttributedFloatList &);

	// Constructs a LogReg classifier 
	virtual PClassifier operator()(PExampleGenerator gen, const int & = 0);
	PClassifier fitModel(PExampleGenerator, const int &, int &, PVariable &);
};



// Logistic regression classifier
// coefficients are needed for each attribute
class ORANGE_API TLogRegClassifier : public TClassifierFD {
public:
	__REGISTER_CLASS

  PDomain continuizedDomain; //P if absent, there is no continuous attributes in original domain
  PEFMDataDescription dataDescription; //P Data needed for classification in presence of undefined values

	// coeficients
	PAttributedFloatList beta; //P estimated beta coefficients for logistic regression
	// beta standard errors
	PAttributedFloatList beta_se; //P estimated standard errors for beta coefficients
	// Wald Z Statistic
	PAttributedFloatList wald_Z; //P Wald Z statstic for beta coefficients
	// P
	PAttributedFloatList P; //P estimated significances for beta coefficients
	// likelihood
	float likelihood; //P Likelihood: The likelihood function is the function which specifies the probability of the sample observed on the basis of a known model, as a function of the model's parameters. 
	// 
	int fit_status; //P Tells how the model fitting ended - either regularly (LogRegFitter.OK), or it was interrupted due to one of beta coefficients escaping towards infinity (LogRegFitter.Infinity) or since the values didn't converge (LogRegFitter.Divergence).

  PImputer imputer; //P if present, it imputes unknown values

	// constructors
	TLogRegClassifier();
	TLogRegClassifier(PDomain);

	virtual PDistribution classDistribution(const TExample &ex);
/*	{ TExample example(domain, ex);
	   example * beta */
};


#endif

//	// error
//	int error; //P Error code thrown by the selected fitter. 0(zero) means that no errors occured while fitting.
//	PVariable error_att; //P Attribute that causes singularity if it occurs. 
