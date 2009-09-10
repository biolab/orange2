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


#ifndef __HEATMAP_HPP
#define __HEATMAP_HPP

#include <vector>
using namespace std;

#include "root.hpp"
#include "orvector.hpp"
#include "px/orangene_globals.hpp"

WRAPPER(ExampleTable)

#define UNKNOWN_F ILLEGAL_FLOAT

void getPercentileInterval(const float *cells, const int &ncells, const float &lowperc, const float &highperc, float &min, float &max);

class ORANGENE_API THeatmap : public TOrange {
public:
  __REGISTER_CLASS
  float *cells;
  float *averages;

  int height; //P bitmap height (in cells)
  int width; //P bitmap width (in cells)

  PExampleTable examples; //PR examples from the whole bitmap
  PIntList exampleIndices; //PR indices to 'examples' (one for row + one at the end)

  THeatmap(const int &h, const int &w, PExampleTable ex);
  ~THeatmap();

  unsigned char *heatmap2string(const int &cellWidth, const int &cellHeight,
                                const int &firstRow, const int &nRows,
                                const float &absLow, const float &absHigh,
                                const float &gamma, bool grid,
                                long &size) const;

  unsigned char *averages2string(const int &cellWidth, const int &cellHeight,
                                 const int &firstRow, const int &nRows,
                                 const float &absLow, const float &absHigh,
                                 const float &gamma, bool grid,
                                 long &size) const;

  float getCellIntensity(const int &y, const int &x) const;
  float getRowIntensity(const int &y) const;

  void getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max) const;
};


OGWRAPPER(Heatmap)

#define THeatmapList TOrangeVector<PHeatmap> 
OGVWRAPPER(HeatmapList)

OGWRAPPER(HeatmapConstructor)

class ORANGENE_API THeatmapConstructor : public TOrange {
public:
  __REGISTER_CLASS

  PExampleTable sortedExamples; //PR sortedExamples

  vector<float *> floatMap; // sorted examples
  vector<int> classBoundaries; // boundaries of classes
  vector<float> lineCenters; // sorted line centers
  vector<float> lineAverages; // sorted line averages
  vector<int> sortIndices; // indices used for sorting the examples

  int nColumns; //PR number of columns
  int nRows; //PR number of rows
  int nClasses; //PR number of classes (0 if the data is not classified)

  THeatmapConstructor(PExampleTable, PHeatmapConstructor baseHeatmap = PHeatmapConstructor(), bool noSorting = false, bool disregardClass=false);
  THeatmapConstructor(); // for pickle
  virtual ~THeatmapConstructor();

  PHeatmapList operator ()(const float &squeeze, float &absLow, float &absHigh);

  unsigned char *getLegend(const int &width, const int &height, const float &gamma, long &size) const;
};

#endif
