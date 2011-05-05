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

#ifndef __PNN_HPP
#define __PNN_HPP

#include "classify.hpp"

WRAPPER(Domain)
WRAPPER(ExampleGenerator)
VWRAPPER(FloatList)

class ORANGE_API TPNN : public TClassifierFD {
public:
  __REGISTER_CLASS

  int dimensions; //PR the number of dimensions
  PFloatList offsets; //P offsets to subtract from the attribute values
  PFloatList normalizers; //P number to divide the values by
  PFloatList averages; //P numbers to use instead of the missing
  bool normalizeExamples; //P if true, attribute values are divided to sum up to 1
  double *bases; // eg x1, y1,  x2, y2,  x3, y3, ... x_dimensions, y_dimensions
  double *radii; // eg sqrt(x1^2+y1^2) ...

  int nExamples; //PR the number of examples
  double *projections; // projections of examples + class
  double minClass, maxClass; //PR the minimal and maximal class value (for regression problems only)

  enum { InverseLinear, InverseSquare, InverseExponential, KNN, Linear };
  int law; //P law

  TPNN(PDomain domain, const int &law = InverseLinear, const bool normalizeExamples = true);
  TPNN(PDomain domain, PExampleGenerator egen, double *bases, const int &law = InverseLinear, const bool normalizeExamples = true);
  TPNN(PDomain, double *, const int &nExamples, double *bases, const int &dimensions, PFloatList off, PFloatList norm, const int &law = InverseLinear, const bool normalizeExamples = true);
  TPNN(PDomain domain, double *examples, const int &nEx, double *ba, const int &dim, PFloatList off, PFloatList norm, const int &law, const vector<int> &attrIndices, int &nOrigRow, const bool normalizeExamples = true);
  TPNN(const int &nDim, const int &nAtt, const int &nEx); // used for pickling: only allocates the memory for the (double *) fields
  TPNN(const TPNN &);
  TPNN &operator =(const TPNN &);

  ~TPNN();

  virtual PDistribution classDistribution(const TExample &);

  virtual void project(const TExample &, double *);
  //virtual void project(double *, double *);
};


class ORANGE_API TP2NN : public TPNN {
public:
  __REGISTER_CLASS

  TP2NN(PDomain domain, PExampleGenerator egen, PFloatList basesX, PFloatList basesY, const int &law = InverseSquare, const bool normalizeExamples = true);
  TP2NN(PDomain, double *projections, const int &nExamples, double *bases, PFloatList off, PFloatList norm, PFloatList avgs, const int &law = InverseSquare, const bool normalizeExamples = true);

  TP2NN(const int &nAttrs, const int &nExamples); // used for pickling: only allocates the memory for the (double *) fields

  virtual TValue operator ()(const TExample &);
  virtual PDistribution classDistribution(const TExample &);

  virtual void classDistribution(const double &, const double &, float *distribution, const int &nClasses) const;
  double averageClass(const double &x, const double &y) const;
  
  virtual void project(const TExample &, double &x, double &y);
  //virtual void project(double *, double *);

  inline void getProjectionForClassification(const TExample &example, double &x, double &y)
  {
    if (example.domain == domain)
      project(example, x, y);
    else {
      TExample nex(domain, example);
      project(nex, x, y);
    }
  }
};

#endif
