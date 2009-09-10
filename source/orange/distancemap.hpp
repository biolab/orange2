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
#include "orvector.hpp"


void ORANGE_API getPercentileInterval(const float *cells, const int &ncells, const float &lowperc, const float &highperc, float &min, float &max);

class ORANGE_API TDistanceMap : public TOrange {
public:
  __REGISTER_CLASS
  float *cells;

  int dim; //PR bitmap dimension (in cells)
  int matrixType; //P 0 lower, 1 upper, 2 symmetric
  PIntList elementIndices; //PR indices to elements (one for row + one at the end)

  TDistanceMap(const int &);
  ~TDistanceMap();

  unsigned char *distanceMap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, bool grid, const int &matrixType, long &size) const;

  float getCellIntensity(const int &y, const int &x) const;
  void getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max);
};


WRAPPER(SymMatrix)
WRAPPER(DistanceMap)

WRAPPER(DistanceMapConstructor)

class ORANGE_API TDistanceMapConstructor : public TOrange {
public:
  __REGISTER_CLASS

  PSymMatrix distanceMatrix; //P distance matrix
  PIntList order; //P order of elements

  TDistanceMapConstructor(PSymMatrix = PSymMatrix());

  PDistanceMap operator ()(const float &squeeze, float &absLow, float &absHigh);
  unsigned char *getLegend(const int &width, const int &height, const float &gamma, long &size) const;
};

#endif
