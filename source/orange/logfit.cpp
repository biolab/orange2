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

#include "logfit.ppp"
#include "converts.hpp"
#include "logreg.hpp"

TLogRegFitter_Cholesky::TLogRegFitter_Cholesky()
{}


// set error values thrown by logistic fitter
const char *TLogRegFitter_Cholesky::errors[] =
    {"LogRegFitter: ngroups < 2, ndf < 0 -- not enough examples with so many attributes",
                   "LogRegFitter: n[i]<0",
                   "LogRegFitter: r[i]<0",
                   "LogRegFitter: r[i]>n[i]: Class has more that 2 values, please use only dichotomous class!",
                   "LogRegFitter: constant variable",
                   "LogRegFitter: singularity",
                   "LogRegFitter: infinity in beta",
                   "LogRegFitter: no convergence" };


// function used only in LogReg fitter, that returns vector length n
// and filled with ones(1)
double *ones(int n) {
    // initialize vector
    double *ret = new double[n];

    // set values
    for (int i=0; i<n; i++) 
        ret[i]=1;
    return ret;
}


PAttributedFloatList TLogRegFitter_Cholesky::operator ()(PExampleGenerator gen, const int &weight, PAttributedFloatList &beta_se, float &likelihood, int &error, PVariable &error_att) {
    // get all needed/necessarily attributes and set
   // check for class variable    
  if (!gen->domain->classVar)
    raiseError("class-less domain");
      // class has to be discrete!
      // if (gen->domain->classVar->varType != TValue::INTVAR)
      //   raiseError("discrete class attribute expected");
      // attributes have to be continuous 
  PITERATE(TVarList, vli, gen->domain->attributes) {
    if ((*vli)->varType == TValue::INTVAR) 
      raiseError("only continuous attributes expected");
  }

  LRInput input;
  LRInfo O;


  // fill input data
  input.data = generateDoubleXMatrix(gen, input.nn, input.k);
  if (gen->domain->classVar->varType == TValue::INTVAR)
    input.success = generateDoubleYVector(gen, weight);
  else
    input.success = generateDoubleYVector_cont(gen, weight);
  //input.trials = ones(input.nn+1);
  input.trials = generateDoubleTrialsVector(gen, weight);

  // initialize output data
  O.nn = input.nn;
  O.k = input.k;
  O.beta = new double[input.k+1];
  O.se_beta = new double[input.k+1];
  O.fit = new double[input.nn+1]; 
  O.stdres = new double[input.nn+1]; 
  O.cov_beta = new double*[input.k+1]; 
  O.dependent = new int[input.k+1];
  int i;
  for(i = 0; i <= input.k; ++i) {
    O.cov_beta[i] = new double[input.k+1];
    O.dependent[i] = 0; // no dependence
  }

    // fit coefficients
  logistic(O.error, input.nn,input.data,input.k,input.success,input.trials,
    O.chisq, O.devnce, O.ndf, O.beta, O.se_beta,
    O.fit, O.cov_beta, O.stdres, O.dependent
  );

  // set error code
  if (O.error == 5)
    error = Constant;
  else if (O.error == 6)
    error = Singularity;
  else if (O.error == 7) 
    error = Infinity;
  else if (O.error == 8)
    error = Divergence;
  else 
    error = OK;

  // get offending attribute
  if (O.error == 6 || O.error == 5 || O.error == 7) {
    int i=1;
    PITERATE(TVarList, vli, gen->domain->attributes) {
      if (O.dependent[i]==1) {
        error_att=*vli;
        break;
      }
      i++;
    }
  }

  if (O.error>0 && O.error<5) {
    raiseError(errors[O.error-1]);
  }

  // create a new domain where class attributes is positioned at the beginning of 
  // the list. I am doing because beta coefficients start with beta0, representing
  // the intercept which is best colligated to class attribute. 
  PVarList enum_attributes = mlnew TVarList(); 
  enum_attributes->push_back(gen->domain->classVar);
  PITERATE(TVarList, vl, gen->domain->attributes) 
    enum_attributes->push_back(*vl);


  // tranfsorm *beta into a PFloatList
  PAttributedFloatList beta=mlnew TAttributedFloatList(enum_attributes);
  beta_se=mlnew TAttributedFloatList(enum_attributes);

  //TODO: obstaja konstruktor, ki pretvori iz navadnega arraya?
  for (i=0; i<input.k+1; i++) {
    beta->push_back(O.beta[i]);
    beta_se->push_back(O.se_beta[i]);
  }

  // Calculate likelihood
  likelihood = - O.devnce; // 

  return beta;
}


double **TLogRegFitter::generateDoubleXMatrix(PExampleGenerator gen, long &numExamples, long &numAttr) {
  double **matrix;
  // get number of instances and allocate number of rows
  numExamples=gen->numberOfExamples();
  numAttr=gen->domain->attributes->size();
  matrix = new double*[numExamples+1];

  { for(int i = 0; i<numExamples; matrix[i++] = NULL); }

  try {
    // copy gen to double matrix
    int n=0;
    matrix[n]= new double[numAttr+1];
    // iteration through examples
    PEITERATE(first, gen) {      
      // row allocation
      matrix[n+1]= new double[numAttr+1];

      int at=0;
      // iteration through attributes
      PITERATE(TVarList, vli, gen->domain->attributes) {
        // copy att. value
        matrix[n+1][at+1]=(*first)[at].floatV;
        at++;
      }
      n++;
    }
  }
  catch (...) {
    for(int i = 0; i<=numExamples; i++)
      if (matrix[i])
        delete matrix[i];
      delete matrix;
  }

  return matrix;
}

double *TLogRegFitter::generateDoubleYVector(PExampleGenerator gen, const int &weightID) {
    // initialize vector
    double *Y = new double[gen->numberOfExamples()+1];
    try {

        // copy gen class to vector *Y
        int n=0;
        PEITERATE(ei, gen) {
            // copy class value
            if (weightID!=0) {
                float weightVal = WEIGHT(*ei);
                Y[n+1]=((float)((*ei).getClass().intV)) * weightVal;
            }
            else
                Y[n+1]=(*ei).getClass().intV;
            n++;
        }
    }
    catch (...) {
        delete Y;
    }

    return Y;
}

double *TLogRegFitter::generateDoubleYVector_cont(PExampleGenerator gen, const int &weightID) {
    // initialize vector
    double *Y = new double[gen->numberOfExamples()+1];
    try {

        // copy gen class to vector *Y
        int n=0;
        PEITERATE(ei, gen) {
            // copy class value
            if (weightID!=0) {
                float weightVal = WEIGHT(*ei);
                Y[n+1]=((float)((*ei).getClass().floatV)) * weightVal;
            }
            else
                Y[n+1]=(*ei).getClass().floatV;
            n++;
        }
    }
    catch (...) {
        delete Y;
    }

    return Y;
}


double *TLogRegFitter::generateDoubleTrialsVector(PExampleGenerator gen, const int &weightID) {
    // initialize vector
    double *T = new double[gen->numberOfExamples()+1];
    try {

        // copy gen class to vector *Y
        int n=0;
        PEITERATE(ei, gen) {
            // copy class value
            if (weightID!=0) {
                T[n+1]=WEIGHT(*ei);
            }
            else
                T[n+1]=1.;
            n++;
        }    
    }
    catch (...) {
      delete T;
      throw;
    }
    return T;
}


LRInput::LRInput() {
    data=NULL;
    success=NULL;
    trials=NULL;
}

LRInput::~LRInput() {
    int i;
    if (data != NULL) {
        for (i=0; i <= nn; ++i)
            delete data[i];
        delete data;
    }
    if (success != NULL) {
        delete success;
    }
}

LRInfo::LRInfo() {
   beta = NULL;        
   se_beta = NULL;    
   fit = NULL;        
   cov_beta = NULL;    
   stdres = NULL;   
   dependent = NULL;
}

LRInfo::~LRInfo() {
    if (cov_beta!=NULL)
        for (int i = 0; i <= k; ++i)
            delete cov_beta[i];
    delete cov_beta;
    delete fit;
    delete beta;
    delete se_beta;
    delete stdres;
    delete dependent;
}

