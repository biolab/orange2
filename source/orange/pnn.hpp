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
  double *bases; // eg x1, y1,  x2, y2,  x3, y3, ... x_dimensions, y_dimensions
  double *radii; // eg sqrt(x1^2+y1^2) ...

  int nExamples; //PR the number of examples
  double *projections; // projections of examples + class

  float exponent2; //P the exponent/2 (eg. -1 for falling with sqr distance)

  TPNN(PDomain domain, const float &exponent2 = -1);
  TPNN(PDomain domain, PExampleGenerator egen, double *bases, const float &exponent2 = -1);
  TPNN(PDomain, double *, const int &nExamples, double *bases, const int &dimensions, PFloatList off, PFloatList norm, const float &exponent2 = -1.0);
  TPNN(PDomain domain, double *examples, const int &nEx, double *ba, const int &dim, PFloatList off, PFloatList norm, const float &e2, const vector<int> &attrIndices, int &nOrigRow);
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

  TP2NN(PDomain domain, PExampleGenerator egen, PFloatList basesX, PFloatList basesY, const float &exponent2 = -1.0);
  TP2NN(PDomain, double *, const int &nExamples, double *bases, PFloatList off, PFloatList norm, const float &exponent2 = -1.0);

  virtual PDistribution classDistribution(const TExample &);
  virtual void classDistribution(const double &, const double &, float *distribution, const int &nClasses) const;

  virtual void project(const TExample &, double &x, double &y);
  //virtual void project(double *, double *);
};

#endif