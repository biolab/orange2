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

#include "distancemap.ppp"

#include "externs.px"

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif


/* Expands the bitmap 
   Each pixel in bitmap 'smmp' is replaced by a square with
     given 'cellWidth' and 'cellHeight'
   The original bitmaps width and height are given by arguments
     'width' and 'height'

   Beside returning the bitmap, the function return its size
   in bytes (argument '&size'). Due to alignment of rows to 4 bytes,
   this does not necessarily equal cellWidth * cellHeight * width * height.
*/

unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight, int &size, float *intensity, const int &width, const int &height, const float &absLow, const float &absHigh, const float &gamma)
{
  const int lineWidth = width * cellWidth;
  const int fill = (4 - lineWidth & 3) & 3;
  const int rowSize = lineWidth + fill;
  size = rowSize * height * cellHeight;

  unsigned char *res = new unsigned char[size];
  unsigned char *resi = res;

  if (gamma == 1.0) {
    const float colorFact = 249.0/(absHigh - absLow);

    for(int line = 0; line<height; line++) {
      unsigned char *thisline = resi;
      int xpoints;
      for(xpoints = width; xpoints--; intensity++) {
        unsigned char col;
        if (*intensity == UNKNOWN_F)
          col = 255;
        else if (*intensity < absLow)
          col = 253;
        else if (*intensity > absHigh)
          col = 254;
        else
          col = int(floor(colorFact * (*intensity - absLow)));

        for(int inpoints = cellWidth; inpoints--; *(resi++) = col);
      }

      resi += fill;
      for(xpoints = cellHeight-1; xpoints--; resi += rowSize)
        memcpy(resi, thisline, lineWidth);
    }
  }
  else {
    const float colorBase = (absLow + absHigh) / 2;
    const float colorFact = 2 / (absHigh - absLow);

    for(int line = 0; line<height; line++) {
      unsigned char *thisline = resi;
      int xpoints;
      for(xpoints = width; xpoints--; intensity++) {
        unsigned char col;

        if (*intensity == UNKNOWN_F)
          col = 255;
        else if (*intensity < absLow)
          col = 253;
        else if (*intensity > absHigh)
          col = 254;
        else {
          float norm = colorFact * (*intensity - colorBase);
          if ((norm > -0.008) && (norm < 0.008))
            norm = 125;
          else
            norm = 124.5 * (1 + (norm<0 ? -exp(gamma * log(-norm)) : exp(gamma * log(norm))));

          if (norm<0)
            col = 0;
          else if (norm>249)
            col = 249;
          else  
            col = int(floor(norm));
        }

        for(int inpoints = cellWidth; inpoints--; *(resi++) = col);
      }

      resi += fill;
      for(xpoints = cellHeight-1; xpoints--; resi += rowSize)
        memcpy(resi, thisline, lineWidth);
    }
  }

  return res;
}



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


unsigned char *TDistanceMap::distancemap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, int &size) const
{
  return bitmap2string(cellWidth, cellHeight, size, cells, dim, dim, absLow, absHigh, gamma);
}


void getPercentileInterval(float *cells, const int &ncells, const float &lowperc, const float &highperc, float &min, float &max);

void TDistanceMap::getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max)
{ getPercentileInterval(cells, dim*dim, lowperc, highperc, min, max); }


float TDistanceMap::getCellIntensity(const int &y, const int &x) const
{ 
  if ((y<0) || (y>=dim))
    raiseError("row index out of range");
  if ((x<0) || (y>=dim))
    raiseError("column index out of range");

  return cells[y*dim+x];
}




TDistanceMapConstructor::TDistanceMapConstructor(PDistanceMatrix m)
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
    push_back(ind);
    inThis = fmod(inThis, toThis);
  }
}

PDistanceMap TDistanceMapConstructor::operator ()(const float &unadjustedSqueeze, float &abslow, float &abshigh)
{
  abshigh = -1e30f;
  abslow = 1e30f;
  
  
  int nLines = int(floor(0.5 + distanceMatrix->dim * unadjustedSqueeze));
  if (!nLines)
    nLines++;
  const float squeeze = float(nLines) / distanceMatrix->dim;
  

  PDistanceMap dm = mlnew TDistanceMap;  
  squeezedIndices = dm->elementIndices.getReference();
  computeSqueezedIndices(nLines, squeeze, squeezedIndices);

  vector<int>::const_iterator sii(squeezedIndices.begin()), sie(squeezedIndices.end());

  float *ri, *fmi = dm->cells;
  int *si, *spec = new int[nLines];

  float *matrixi = distanceMatrix->elements;

  float inThisRow = 0;
  int xpoint;

  for(int line = 0; line<nLines; cnt--; inThisRow-=1.0, ami++, fmi+=nLines) {
    for(xpoint = nColumns, ri = fmi, si = spec; xpoint--; *(ri++) = 0.0, *(si++) = 0);

    for(; (line < nLines) && (inThisRow < 1.0); inThisRow += squeeze, line++) {
      for(xpoint = 0, ri = fmi, si = spec; xpoint--; fri++, ri++, si++)
          if (*fri != UNKNOWN_F) {
            *ri += *fri;
            (*si)++;
          }

      }

      hm->exampleIndices->push_back(exampleIndex);
      for(xpoint = nColumns, si = spec, ri = fmi; xpoint--; ri++, si++) {
        if (*si) {
          *ri = *ri / *si;
          if (*ri < abslow)
            abslow = *ri;
          if (*ri > abshigh)
            abshigh = *ri;
        }
        else
          *ri = UNKNOWN_F;
      }

    }
  }

  delete spec;

  return hml;
}


unsigned char *THeatmapConstructor::getLegend(const int &width, const int &height, const float &gamma, int &size) const
{
  float *fmp = new float[width], *fmpi = fmp;

  float wi1 = width-1;
  for(int wi = 0; wi<width; *(fmpi++) = (wi++)/wi1);
  
  unsigned char *legend = bitmap2string(1, height, size, fmp, width, 1, 0, 1, gamma);
  delete fmp;
  return legend;
}
