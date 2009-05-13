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


#ifndef __SYMMATRIX_HPP
#define __SYMMATRIX_HPP

#include "root.hpp"
#include <vector>
#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <algorithm>

using namespace std;

class ORANGE_API TSymMatrix : public TOrange
{
public:
  __REGISTER_CLASS

  CLASSCONSTANTS(Shape) enum { Lower, Upper, Symmetric, LowerFilled, UpperFilled };

  int dim; //PR matrix dimension
  int matrixType; //P(&SymMatrix_Shape) 0 lower, 1 upper, 2 symmetric, 3 lower_filled, 4 upper_filled

  float *elements;

  TSymMatrix(const int &adim, const float &init = 0)
  : dim(adim),
    matrixType(Symmetric),
    elements(mlnew float[((adim+1)*(adim+2))>>1])
  { for(float *pi = elements, *pe = elements+(((adim+1)*(adim+2))>>1); pi!=pe; *(pi++) = init); }

  ~TSymMatrix()
  { mldelete elements; }

  int getindex(const int &i, const int &j, bool raiseExceptions = true) const;

  inline float &getref(const int &i, const int &j)
  { return elements[getindex(i, j)]; }

  inline const float &getref(const int &i, const int &j) const
  { return elements[getindex(i, j)]; }

  inline const float getitem(const int &i, const int &j) const
  { const int index = getindex(i, j, false);
    return index<0 ? float(0.0) : elements[getindex(i, j)];
  }

  typedef std::pair<int, double> coord_t;
  struct pkt_less {
      bool operator ()(const coord_t &e1, const coord_t &e2) const {
    	  return (e1.second < e2.second);
      }
  };

  void getknn(const int &i, const int &k, vector<int> &knn) {
	  vector<coord_t> knn_tmp;
	  int j;
	  for (j=0; j < dim; j++)
		  if (j != i)
		  	knn_tmp.push_back(coord_t(j, elements[getindex(i, j)]));

		sort(knn_tmp.begin(), knn_tmp.end(), pkt_less());

		for (j=0; j < k; j++) {
			knn.push_back(knn_tmp[j].first);
		}
  }

  void index2coordinates(const float *f, int &x, int &y) const
  { index2coordinates(f-elements, x, y); }

  static void index2coordinates(const int &index, int &x, int &y)
  {
    x = int(floor( (sqrt(float(1+8*index)) -1) / 2));
    y = index - (x*(x+1))/2;
  }
};

WRAPPER(SymMatrix)

#endif
