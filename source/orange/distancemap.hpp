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


#ifndef __DISTANCEMAP_HPP
#define __DISTANCEMAP_HPP

#include "root.hpp"

WRAPPER(DistanceMatrix)

#define UNKNOWN_F -1e30f

class TDistanceMap : public TOrange {
public:
  __REGISTER_CLASS
  float *cells;

  int dim; //P bitmap dimension (in cells)
  int matrixType; //P 0 lower, 1 upper, 2 symmetric
  PIntList elementIndices; //PR indices to elements (one for row + one at the end)

  TDistanceMap(const int &);
  ~TDistanceMap();

  unsigned char *distancemap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, int &size) const;

  float getCellIntensity(const int &y, const int &x) const;
  void getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max);
};


WRAPPER(SymMatrix)

WRAPPER(DistanceMapConstructor)

class TDistanceMapConstructor : public TOrange {
public:
  __REGISTER_CLASS

  PSymMatrix distanceMatrix;

  TDistanceMapConstructor(PSymMatrix);

  PDistanceMap operator ()(const float &squeeze, float &absLow, float &absHigh);
  unsigned char *getLegend(const int &width, const int &height, const float &gamma, int &size) const;
};

#endif