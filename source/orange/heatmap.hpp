#ifndef __HEATMAP_HPP
#define __HEATMAP_HPP

#include "root.hpp"

WRAPPER(ExampleTable)

#define UNKNOWN_F -1e30f

unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight, int &size, float *bmp, const int &wdth, const int &height, const float &absLow, const float &absHigh, const float &gamma);

class THeatmap : public TOrange {
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

  unsigned char *THeatmap::heatmap2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, int &size) const;
  unsigned char *THeatmap::averages2string(const int &cellWidth, const int &cellHeight, const float &absLow, const float &absHigh, const float &gamma, int &size) const;

  float getCellIntensity(const int &y, const int &x) const;
  float getRowIntensity(const int &y) const;

  void getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max);
};


WRAPPER(Heatmap)

#define THeatmapList TOrangeVector<PHeatmap> 
VWRAPPER(HeatmapList)

WRAPPER(HeatmapConstructor)

class THeatmapConstructor : public TOrange {
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
  virtual ~THeatmapConstructor();

  PHeatmapList operator ()(const float &squeeze, float &absLow, float &absHigh);

  unsigned char *getLegend(const int &width, const int &height, const float &gamma, int &size) const;
};

#endif