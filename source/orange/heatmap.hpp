#ifndef __HEATMAP_HPP
#define __HEATMAP_HPP

#include "root.hpp"

WRAPPER(ExampleTable)

#define UNKNOWN_F -1e30f

unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight, int &size, unsigned char *bmp, const int &wdth, const int &height);

class THeatmap : public TOrange {
public:
  __REGISTER_CLASS
  unsigned char *bitmap;
  unsigned char *averages;

  int height; //P bitmap height (in cells)
  int width; //P bitmap width (in cells)

  PExampleTable examples; //PR examples from the whole bitmap
  PIntList exampleIndices; //PR indices to 'examples' (one for row + one at the end)

  THeatmap(const int &h, const int &w, PExampleTable ex);
  ~THeatmap();

  unsigned char *heatmap2string(const int &cellWidth, const int &cellHeight, int &size) {
    return bitmap2string(cellWidth, cellHeight, size, bitmap, width, height);
  }

  unsigned char *averages2string(const int &cellWidth, const int &cellHeight, int &size) {
    return bitmap2string(cellWidth, cellHeight, size, averages, 1, height);
  }
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

  float absLow; //PR the lowest intensity that has appeared in the last rendering
  float absHigh; //PR the highest intensity that has appeared in the last rendering

  float gamma; //PR the gamma of the last rendering
  float colorBase; //PR the intensity that corresponds to the lowest color (function depends upon gamma!)
  float colorFact; //PR color scaling factor (function depends upon gamma!)

  THeatmapConstructor(PExampleTable, PHeatmapConstructor baseHeatmap = PHeatmapConstructor(), bool noSorting = false);
  virtual ~THeatmapConstructor();

  PHeatmapList operator ()(const float &squeeze, const float &lowerBound, const float &upperBound, const float &gamma);

  unsigned char *getLegend(const int &width, const int &height, int &size) const;
 
  inline int computePixel(float intensity) const
  {
    if (intensity == UNKNOWN_F)
      return 255;
    if (intensity < absLow)
      return 253;
    if (intensity > absHigh)
      return 254;

    float norm = colorFact * (intensity - colorBase);
    if ((norm > -0.008) && (norm < 0.008))
      norm = 125;
    else
      norm = 124.5 * (1 + (norm<0 ? -exp(gamma * log(-norm)) : exp(gamma * log(norm))));

    if (norm<0)
      return 0;
    if (norm>249)
      return 249;
    return int(floor(norm));
  }

  inline int computePixelGamma1(float intensity) const
  {
    if (intensity == UNKNOWN_F)
      return 255;
    if (intensity < absLow)
      return 253;
    if (intensity > absHigh)
      return 254;
    if (intensity < colorBase)
      return 0;

    const float norm = colorFact * (intensity - colorBase);
    if (norm>249)
      return 249;
    return int(floor(norm));
  }
};

#endif