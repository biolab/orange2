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


#include <algorithm>
#include <queue>
#include "module.hpp"
#include "errors.hpp"
#include "symmatrix.hpp"

#include "distancemap.ppp"

#include "externs.px"

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif


unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight, int &size, float *intensity, const int &width, const int &height, const float &absLow, const float &absHigh, const float &gamma, bool grid = true);

#define UNKNOWN_F -1e30f

TDistanceMap::TDistanceMap(const int &d)
: cells(new float [d*d]),
  dim(d),
  elementIndices(new TIntList)
{}


TDistanceMap::~TDistanceMap()
{
  delete cells;
}


unsigned char *TDistanceMap::distanceMap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, bool grid, int &size) const
{
  return bitmap2string(cellWidth, cellHeight, size, cells, dim, dim, absLow, absHigh, gamma, grid);
}


void getPercentileInterval(const float *cells, const int &ncells, const float &lowperc, const float &highperc, float &min, float &max);

void TDistanceMap::getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max)
{ ::getPercentileInterval(cells, dim*dim, lowperc, highperc, min, max); }


float TDistanceMap::getCellIntensity(const int &y, const int &x) const
{ 
  if ((y<0) || (y>=dim))
    raiseError("row index out of range");
  if ((x<0) || (y>=dim))
    raiseError("column index out of range");

  return cells[y*dim+x];
}




TDistanceMapConstructor::TDistanceMapConstructor(PSymMatrix m)
: distanceMatrix(m)
{}


void computeSqueezedIndices(const int &n, const float &squeeze, vector<int> &indices)
{
  float inThis = 0;
  int ind = 0;
  indices.push_back(ind);
  while(ind<n) {
    float toThis = (1.0 - inThis) / squeeze;
    ind += floor(toThis);
    indices.push_back(ind);
    inThis = fmod(inThis, toThis);
  }
}

#define UPDATE_LOW_HIGH if (incell > abshigh) abshigh = incell; if (incell < abslow) abslow = incell;

PDistanceMap TDistanceMapConstructor::operator ()(const float &unadjustedSqueeze, float &abslow, float &abshigh)
{
  checkProperty(distanceMatrix);

  const TSymMatrix &distMat = distanceMatrix.getReference();
  abshigh = -1e30f;
  abslow = 1e30f;
  
  if (order) {
    if (order->size() != distMat.dim)
      raiseError("size of 'order' does not match the size of the distance matrix");

    if (unadjustedSqueeze < 1.0 - 1e-5) {
      int nLines = int(floor(0.5 + order->size() * unadjustedSqueeze));
      if (!nLines)
        nLines++;
      const float squeeze = float(nLines) / order->size();
  
      PIntList psqi = new TIntList();
      vector<int> &squeezedIndices = psqi->__orvector;
      computeSqueezedIndices(nLines, squeeze, squeezedIndices);

      nLines = squeezedIndices.size() - 1;

      PDistanceMap dm = mlnew TDistanceMap(nLines);
      dm->elementIndices = psqi;

      vector<int>::const_iterator row_squeezei(squeezedIndices.begin()), row_squeezen(row_squeezei+1), squeezee(squeezedIndices.end());
      for(int row = 0; row_squeezen != squeezee; row_squeezei++, row_squeezen++, row++) {
        // this needs to be float (to avoid int division)
        float row_elements = *row_squeezen - *row_squeezei;

        // to diagonal, exclusive
        vector<int>::const_iterator col_squeezei(squeezedIndices.begin()), col_squeezen(col_squeezei+1);
        for(int column = 0; col_squeezei != row_squeezei; col_squeezei++, col_squeezen++, column++) {

          float incell = 0.0;
          vector<int>::const_iterator row_orderi(order->begin()+*row_squeezei), row_ordere(order->begin()+*row_squeezen);
          for(; row_orderi != row_ordere; row_orderi++) {
            vector<int>::const_iterator col_orderi(order->begin()+*col_squeezei), col_ordere(order->begin()+*col_squeezen);
            for(; col_orderi != col_ordere; col_orderi++)
              incell += distMat.getitem(*row_orderi, *col_orderi);
          }
      
          incell /= row_elements * (*col_squeezen - *col_squeezei);
          dm->cells[row*nLines+column] = dm->cells[column*nLines+row] = incell;
          UPDATE_LOW_HIGH;
        }

        // diagonal
        {
          float incell = 0.0;
          vector<int>::const_iterator row_orderi(order->begin()+*row_squeezei), row_ordere(order->begin()+*row_squeezen);
          for(; row_orderi != row_ordere; row_orderi++) {
            vector<int>::const_iterator col_orderi(order->begin()+*row_squeezei);
            for(; col_orderi != row_orderi; col_orderi++)
              incell += distMat.getitem(*row_orderi, *col_orderi);
            incell += distMat.getitem(*row_orderi, *row_orderi);
          }
      
          incell /= row_elements * (row_elements+1) / 2.0;
          dm->cells[row*(nLines+1)] = incell;
          UPDATE_LOW_HIGH;
        }
      }
  
      return dm;
    }

    else { // order && no squeeze
      const int &dim = distMat.dim;
      PDistanceMap dm = mlnew TDistanceMap(dim);

      vector<int>::const_iterator row_orderi(order->begin()), row_ordere(order->end());
      for(int row = 0; row_orderi != row_ordere; row_orderi++, row++) {
        vector<int>::const_iterator col_orderi(order->begin());
        for(int column = 0; col_orderi != row_orderi; col_orderi++, column++) {
          const float &incell = distMat.getref(*row_orderi, *col_orderi);
          dm->cells[row*dim+column] = dm->cells[column*dim+row] = incell;
          UPDATE_LOW_HIGH;
        }
        const float &incell = distMat.getref(*row_orderi, *row_orderi);
        dm->cells[row*(dim+1)] = incell;
        UPDATE_LOW_HIGH;
      }

      return dm;
    }
  }

  else { // no order
    if (unadjustedSqueeze < 1 - 1e-5) {
      int nLines = int(floor(0.5 + distMat.dim * unadjustedSqueeze));
      if (!nLines)
        nLines++;
      const float squeeze = float(nLines) / distMat.dim;
  
      PIntList psqi = new TIntList();
      vector<int> &squeezedIndices = psqi->__orvector;
      computeSqueezedIndices(nLines, squeeze, squeezedIndices);

      nLines = squeezedIndices.size() - 1;

      PDistanceMap dm = mlnew TDistanceMap(nLines);
      dm->elementIndices = psqi;

      vector<int>::const_iterator row_squeezei(squeezedIndices.begin()), row_squeezen(row_squeezei+1), squeezee(squeezedIndices.end());
      for(int row = 0; row_squeezen != squeezee; row_squeezei++, row_squeezen++, row++) {
        // this needs to be float (to avoid int division)
        float row_elements = *row_squeezen - *row_squeezei;

        // to diagonal, exclusive
        vector<int>::const_iterator col_squeezei(squeezedIndices.begin()), col_squeezen(col_squeezei+1);
        for(int column = 0; col_squeezei != row_squeezei; col_squeezei++, col_squeezen++, column++) {
          int col_elements = *col_squeezen - *col_squeezei;

          float incell = 0.0;
          int origrow = *row_squeezei, origrowe = *row_squeezen;
          for(; origrow != origrowe; origrow++) {
            const float *origval = &distMat.getref(origrow, *col_squeezei);
            for(int ce = col_elements; ce--; incell += *(origval++));
          }
      
          incell /= row_elements * col_elements;
          dm->cells[row*nLines+column] = dm->cells[column*nLines+row] = incell;
          UPDATE_LOW_HIGH;
        }

        // diagonal
        {
          float incell = 0.0;
          int origrow = *row_squeezei, origrowe = *row_squeezen;
          for(; origrow != origrowe; origrow++) {
            const float *origval = &distMat.getref(origrow, *col_squeezei);
            for(int ce = origrow - *row_squeezei+1; ce--; incell += *(origval++));
          }
      
          incell /= row_elements * (row_elements+1) / 2.0;
          dm->cells[row*(nLines+1)] = incell;
          UPDATE_LOW_HIGH;
        }
      }
  
      return dm;

    }

    else {// no order && no squeeze
      const int &dim = distMat.dim;
      PDistanceMap dm = mlnew TDistanceMap(dim);
      for(int row = 0; row < dim; row++) {
        for(int column = 0; column < row; column++) {
          const float &incell = distMat.getref(row, column);
          dm->cells[row*dim+column] = dm->cells[column*dim+row] = incell;
          UPDATE_LOW_HIGH;
        }
        const float &incell = distMat.getref(row, row);
        dm->cells[row*(dim+1)] = incell;
        UPDATE_LOW_HIGH;
      }

      return dm;
    }
  }
}


unsigned char *TDistanceMapConstructor::getLegend(const int &width, const int &height, const float &gamma, int &size) const
{
  float *fmp = new float[width], *fmpi = fmp;

  float wi1 = width-1;
  for(int wi = 0; wi<width; *(fmpi++) = (wi++)/wi1);
  
  unsigned char *legend = bitmap2string(1, height, size, fmp, width, 1, 0, 1, gamma);
  delete fmp;
  return legend;
}
