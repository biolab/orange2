#ifndef __HEATMAP_HPP
#define __HEATMAP_HPP

#include "root.hpp"

WRAPPER(ExampleTable)

class THeatmap : public TOrange {
public:
  __REGISTER_CLASS
  unsigned char *bitmap;

  int height; //P bitmap height (in cells)
  int width; //P bitmap width (in cells)

  PExampleTable examples; //PR examples from the whole bitmap
  PIntList exampleIndices; //PR indices to 'examples' (one for row + one at the end)

  THeatmap(const int &h, const int &w, PExampleTable ex);
  ~THeatmap();

  unsigned char *heatmap2string(const int &cellWidth, const int &cellHeight, int &size);
};


WRAPPER(Heatmap)

#define THeatmapList TOrangeVector<PHeatmap> 
VWRAPPER(HeatmapList)


class THeatmapConstructor : public TOrange {
public:
  __REGISTER_CLASS

  PExampleTable sortedExamples; //PR sortedExamples

  vector<float *> floatMap; // for sorted examples
  vector<int> classBoundaries;
  vector<float> lineCenters; // sorted line centers

  int nColumns; //PR number of columns
  int nRows; //PR number of rows
  int nClasses; //PR number of classes (0 if the data is not classified)

  float absLow; //PR the lowest intensite that has appeared in the last rendering
  float absHigh; //PR the highest intensiti that has appeared in the last rendering

  float gamma; //PR the gamma of the last rendering
  float colorBase; //PR the intensity that corresponds to the lowest color (function depends upon gamma!)
  float colorFact; //PR color scaling factor (function depends upon gamma!)

  THeatmapConstructor(PExampleTable);
  virtual ~THeatmapConstructor();

  PHeatmapList operator ()(const float &squeeze, const float &lowerBound, const float &upperBound, const float &gamma);
};

WRAPPER(HeatmapConstructor);

#endif