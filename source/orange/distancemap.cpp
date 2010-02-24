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


#include <algorithm>
#include <queue>
#include "orange.hpp"
#include "symmatrix.hpp"

#include "distancemap.ppp"

#include "externs.px"

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif


#define UNKNOWN_F ILLEGAL_FLOAT

#define PAINT_PIXEL_LINE                     \
{ unsigned char col;                         \
  if (*intensity == UNKNOWN_F)   col = 255;  \
  else if (*intensity < absLow)  col = 253;  \
  else if (*intensity > absHigh) col = 254;  \
  else \
    if (nogamma) col = int(floor(colorFact * (*intensity - absLow))); \
    else { \
      float norm = colorFact * (*intensity - colorBase); \
      if ((norm > -0.008) && (norm < 0.008)) norm = 125; \
      else norm = 124.5 * (1 + (norm<0 ? -exp(gamma * log(-norm)) : exp(gamma * log(norm)))); \
\
      if (norm<0)        col = 0; \
      else if (norm>249) col = 249; \
      else col = int(floor(norm)); \
   } \
\
  for(int inpoints = cellWidth; inpoints--; *(resi++) = col); \
  if (grid) \
    resi[-1] = 252; \
}

/* This function was originally stolen from heatmap.cpp, but modified to
   handle triangular matrices */

unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight, long &size,
                             float *intensity, const int &width, const int &height,
                             const float &absLow, const float &absHigh, const float &gamma,
                             bool grid, const int &matrixType)
{
  const int lineWidth = width * cellWidth;
  const int fill = (4 - lineWidth & 3) & 3;
  const int rowSize = lineWidth + fill;
  size = rowSize * height * cellHeight;

  unsigned char *res = new unsigned char[size];
  if (!res)
    raiseErrorWho("bitmap2string", "not enough memory (%i bytes) for the bitmap", size);

  unsigned char *resi = res;

  if (grid && ((cellHeight<3) || (cellWidth < 3)))
    grid = false;

  bool nogamma;
  float colorFact, colorBase;
  if (gamma == 1.0) {
    nogamma = true;
    colorFact = 249.0/(absHigh - absLow);
  }
  else {
    nogamma = false;
    colorBase = (absLow + absHigh) / 2;
    colorFact = 2 / (absHigh - absLow);
  }

  for(int line = 0; line<height; line++) {
    int xpoints;

    unsigned char *thisline = resi;

    switch(matrixType) {
      case 0: // lower
        for(xpoints = line+1; xpoints--; intensity++)
          PAINT_PIXEL_LINE;

        memset(resi, 251, (width-line-1)*cellWidth);
        resi += (width-line-1)*cellWidth;
        intensity += width-line-1;
        break;

      case 1: // upper
        memset(resi, 251, line*cellWidth);
        resi += line*cellWidth;
        intensity += line;

        for(xpoints = width-line; xpoints--; intensity++)
          PAINT_PIXEL_LINE;
        break;

      case 2:
      default:
        for(xpoints = width; xpoints--; intensity++)
          PAINT_PIXEL_LINE;
        break;
    }

    resi += fill;
    for(xpoints = grid ? cellHeight-2 : cellHeight-1; xpoints--; resi += rowSize)
      memcpy(resi, thisline, lineWidth);
      
    if (grid) {
      memset(resi, 252, rowSize);
      resi += rowSize;
      
      unsigned char *bi;
      if (height >= 10) {
        for(bi = thisline, xpoints = width; xpoints--; bi += cellWidth)
          bi[0] = bi[1] = bi[cellWidth-3] = bi[cellWidth-2] = bi[rowSize] = bi[rowSize+cellWidth-2] = 252;
        for(bi = thisline+rowSize*(cellHeight-2), xpoints = width; xpoints--; bi += cellWidth)
          bi[0] = bi[1] = bi[cellWidth-3] = bi[cellWidth-2] = bi[-rowSize] = bi[-rowSize+cellWidth-2] = 252;
      }
      else {
        for(bi = thisline, xpoints = width; xpoints--; bi += cellWidth)
          bi[0] = bi[cellWidth-2] = 252;
        for(bi = thisline+rowSize*(cellHeight-2), xpoints = width; xpoints--; bi += cellWidth)
          bi[0] = bi[cellWidth-2] = 252;
      }
    }
  }

  return res;
}


TDistanceMap::TDistanceMap(const int &d)
: cells(new float [d*d]),
  dim(d),
  elementIndices(new TIntList)
{}


TDistanceMap::~TDistanceMap()
{
  delete cells;
}


unsigned char *TDistanceMap::distanceMap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, bool grid, const int &matrixType, long &size) const
{
  return bitmap2string(cellWidth, cellHeight, size, cells, dim, dim, absLow, absHigh, gamma, grid, matrixType);
}


void getPercentileInterval(const float *cells, const int &ncells, const float &lowperc, const float &highperc, float &min, float &max)
{
  const int nlow = lowperc * ncells;
  const int nhigh = highperc * ncells;

  priority_queue<float, vector<float>, greater<float> > lower;
  priority_queue<float, vector<float>, less<float> > upper;

  int i = ncells;
  for(const float *ci = cells; i--; ci++) {
    lower.push(*ci);
    if (lower.size() > nlow)
      lower.pop();
    upper.push(*ci);
    if (upper.size() > nhigh)
      upper.pop();
  }

  min = lower.top();
  max = upper.top();
}


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


void computeSqueezedIndices(const int &origLines, const int &squeezedLines, TIntList &indices)
{
  float k = float(origLines) / squeezedLines;
  for(int i = 0; i <= squeezedLines; i++)
    indices.push_back(floor(0.5+i*k));
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
  
      PIntList psqi = new TIntList();
      TOrangeVector<int, false> &squeezedIndices = psqi.getReference();
      computeSqueezedIndices(distMat.dim, nLines, squeezedIndices);

      nLines = squeezedIndices.size() - 1;

      PDistanceMap dm = mlnew TDistanceMap(nLines);
      dm->elementIndices = psqi;

      TIntList::const_iterator row_squeezei(squeezedIndices.begin()), row_squeezen(row_squeezei+1), squeezee(squeezedIndices.end());
      for(int row = 0; row_squeezen != squeezee; row_squeezei++, row_squeezen++, row++) {
        // this needs to be float (to avoid int division)
        float row_elements = *row_squeezen - *row_squeezei;

        // to diagonal, exclusive
        TIntList::const_iterator col_squeezei(squeezedIndices.begin()), col_squeezen(col_squeezei+1);
        for(int column = 0; col_squeezei != row_squeezei; col_squeezei++, col_squeezen++, column++) {

          float incell = 0.0;
          TIntList::const_iterator row_orderi(order->begin()+*row_squeezei), row_ordere(order->begin()+*row_squeezen);
          for(; row_orderi != row_ordere; row_orderi++) {
            TIntList::const_iterator col_orderi(order->begin()+*col_squeezei), col_ordere(order->begin()+*col_squeezen);
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
          TIntList::const_iterator row_orderi(order->begin()+*row_squeezei), row_ordere(order->begin()+*row_squeezen);
          for(; row_orderi != row_ordere; row_orderi++) {
            TIntList::const_iterator col_orderi(order->begin()+*row_squeezei);
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

      TIntList::const_iterator row_orderi(order->begin()), row_ordere(order->end());
      for(int row = 0; row_orderi != row_ordere; row_orderi++, row++) {
        TIntList::const_iterator col_orderi(order->begin());
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
  
      PIntList psqi = new TIntList();
      TOrangeVector<int, false> &squeezedIndices = psqi.getReference();
      computeSqueezedIndices(distMat.dim, nLines, squeezedIndices);

      nLines = squeezedIndices.size() - 1;

      PDistanceMap dm = mlnew TDistanceMap(nLines);
      dm->elementIndices = psqi;

      TIntList::const_iterator row_squeezei(squeezedIndices.begin()), row_squeezen(row_squeezei+1), squeezee(squeezedIndices.end());
      for(int row = 0; row_squeezen != squeezee; row_squeezei++, row_squeezen++, row++) {
        // this needs to be float (to avoid int division)
        float row_elements = *row_squeezen - *row_squeezei;

        // to diagonal, exclusive
        TIntList::const_iterator col_squeezei(squeezedIndices.begin()), col_squeezen(col_squeezei+1);
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
      dm->elementIndices = new TIntList();
      TOrangeVector<int, false> &squeezedIndices = dm->elementIndices.getReference();

      for(int row = 0; row < dim; row++) {
        squeezedIndices.push_back(row);
        for(int column = 0; column < row; column++) {
          const float &incell = distMat.getref(row, column);
          dm->cells[row*dim+column] = dm->cells[column*dim+row] = incell;
          UPDATE_LOW_HIGH;
        }
        const float &incell = distMat.getref(row, row);
        dm->cells[row*(dim+1)] = incell;
        UPDATE_LOW_HIGH;
      }

      squeezedIndices.push_back(dim);

      return dm;
    }
  }
}


unsigned char *TDistanceMapConstructor::getLegend(const int &width, const int &height, const float &gamma, long &size) const
{
  float *fmp = new float[width], *fmpi = fmp;

  float wi1 = width-1;
  for(int wi = 0; wi<width; *(fmpi++) = (wi++)/wi1);
  
  unsigned char *legend = bitmap2string(1, height, size, fmp, width, 1, 0, 1, gamma, false, 2);
  delete fmp;
  return legend;
}
