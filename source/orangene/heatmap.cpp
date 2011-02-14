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
#include "table.hpp"
//#include "module.hpp"
#include "cls_orange.hpp"
#include "cls_example.hpp"

#include "ppp/heatmap.ppp"

#include "px/externs.px"

#ifdef _MSC_VER
  #pragma warning (disable : 4786 4114 4018 4267 4244)
#endif

DEFINE_TOrangeVector_classDescription(PHeatmap, "THeatmapList", true, ORANGENE_API)

/* Expands the bitmap 
   Each pixel in bitmap 'smmp' is replaced by a square with
     given 'cellWidth' and 'cellHeight'
   The original bitmaps width and height are given by arguments
     'width' and 'height'

   Beside returning the bitmap, the function return its size
   in bytes (argument '&size'). Due to alignment of rows to 4 bytes,
   this does not necessarily equal cellWidth * cellHeight * width * height.
*/

unsigned char *bitmap2string(const int &cellWidth, const int &cellHeight,
                             const int &firstRow, const int &nRows,
                             long &size,
                             float *intensity, const int &width, const int &height,
                             const float &absLow, const float &absHigh, const float &gamma,
                             bool grid)
{
  const int lineWidth = width * cellWidth;
  const int fill = (4 - lineWidth & 3) & 3;
  const int rowSize = lineWidth + fill;
  size = rowSize * nRows * cellHeight;

  unsigned char *res = new unsigned char[size];
  unsigned char *resi = res;

  if (grid && ((cellHeight<3) || (cellWidth < 3)))
    grid = false;

  int line = firstRow;
  int lline = firstRow + nRows;
  intensity += firstRow * width;

  if (gamma == 1.0) {
    const float colorFact = 249.0/(absHigh - absLow);

    for(; line<lline; line++) {
      int xpoints;

      unsigned char *thisline = resi;
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
        if (grid)
          resi[-1] = 252;
      }

      resi += fill;
      for(xpoints = grid ? cellHeight-2 : cellHeight-1; xpoints--; resi += rowSize)
        memcpy(resi, thisline, lineWidth);

      if (grid) {
        memset(resi, 252, rowSize);
        resi += rowSize;
      }
    }
  }
  else {
    const float colorBase = (absLow + absHigh) / 2;
    const float colorFact = 2 / (absHigh - absLow);

    for(; line<lline; line++) {
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
        if (grid)
          resi[-1] = 252;
      }

      resi += fill;
      for(xpoints = grid ? cellHeight-2 : cellHeight-1; xpoints--; resi += rowSize)
        memcpy(resi, thisline, lineWidth);

      if (grid) {
        memset(resi, 252, rowSize);
        resi += rowSize;
      }
    }
  }

  return res;
}



class CompareIndicesWClass {
public:
  const vector<float> &centers;
  const vector<int> &classes;

  CompareIndicesWClass(const vector<float> &acen, const vector<int> &acl)
  : centers(acen),
    classes(acl)
  {};

  bool operator() (const int &i1, const int &i2)
  { return    (classes[i1]<classes[i2])
           || ((classes[i1] == classes[i2]) && (centers[i1] < centers[i2])); }
};


class CompareIndicesClass {
public:
  const vector<int> &classes;

  CompareIndicesClass(const vector<int> &acl)
  : classes(acl)
  {};

  bool operator() (const int &i1, const int &i2)
  { return classes[i1]<classes[i2]; }
};


class CompareIndices {
public:
  const vector<float> &centers;

  CompareIndices(const vector<float> &acen)
  : centers(acen)
  {};

  bool operator() (const int &i1, const int &i2)
  { return centers[i1] < centers[i2]; }
};


WRAPPER(ExampleTable);


THeatmap::THeatmap(const int &h, const int &w, PExampleTable ex)
: cells(new float [h*w]),
  averages(new float [h]),
  height(h),
  width(w),
  examples(ex),
  exampleIndices(new TIntList())
{}


THeatmap::~THeatmap()
{
  delete cells;
  delete averages;
}


unsigned char *THeatmap::heatmap2string(const int &cellWidth, const int &cellHeight, const int &firstRow, const int &nRows, const float &absLow, const float &absHigh, const float &gamma, bool grid, long &size) const
{
  return bitmap2string(cellWidth, cellHeight, firstRow, nRows, size, cells, width, height, absLow, absHigh, gamma, grid);
}

unsigned char *THeatmap::averages2string(const int &cellWidth, const int &cellHeight, const int &firstRow, const int &nRows, const float &absLow, const float &absHigh, const float &gamma, bool grid, long &size) const
{
  return bitmap2string(cellWidth, cellHeight, firstRow, nRows, size, averages, 1, height, absLow, absHigh, gamma, grid);
}


float THeatmap::getCellIntensity(const int &y, const int &x) const
{ 
  if ((y<0) || (y>=height))
    raiseError("row index out of range");
  if ((x<0) || (y>=height))
    raiseError("column index out of range");

  return cells[y*width+x];
}


float THeatmap::getRowIntensity(const int &y) const
{ 
  if ((y<0) || (y>=height))
    raiseError("row index out of range");

  return averages[y];
}


void THeatmap::getPercentileInterval(const float &lowperc, const float &highperc, float &min, float &max) const
{ ::getPercentileInterval(cells, width*height, lowperc, highperc, min, max); }


THeatmapConstructor::THeatmapConstructor()
{}


THeatmapConstructor::THeatmapConstructor(PExampleTable table, PHeatmapConstructor baseHeatmap, bool noSorting, bool disregardClass)
: sortedExamples(new TExampleTable(table, 1)), // lock, but do not copy
  floatMap(),
  classBoundaries(),
  lineCenters(),
  nColumns(table->domain->attributes->size()),
  nRows(table->numberOfExamples()),
  nClasses(0)
{
  TExampleTable &etable = table.getReference();
  if (baseHeatmap && (etable.numberOfExamples() != baseHeatmap->sortedExamples->numberOfExamples()))
    raiseError("'baseHeatmap has a different number of spots");

  TExampleTable &esorted = sortedExamples.getReference();

  bool haveBase = baseHeatmap;

  PITERATE(TVarList, ai, etable.domain->attributes)
    if ((*ai)->varType != TValue::FLOATVAR)
      raiseError("data contains a discrete attribute '%s'", (*ai)->get_name().c_str());

  if (etable.domain->classVar && !disregardClass) {
    if (etable.domain->classVar->varType != TValue::INTVAR)
      raiseError("class attribute is not discrete");
    nClasses = etable.domain->classVar->noOfValues();
    if (!haveBase)
      for(int i = nClasses+1; i; i--)
        classBoundaries.push_back(0);
  }
  else {
    nClasses = 0;
    classBoundaries.push_back(0);
    classBoundaries.push_back(nRows);
  }

  vector<float *> tempFloatMap;
  vector<float> tempLineCenters;
  vector<float> tempLineAverages;
  vector<int>classes;

  tempFloatMap.reserve(nRows);

  tempLineCenters.reserve(nRows);
  if (!haveBase)
    sortIndices.reserve(nRows);

  bool pushSortIndices = !haveBase && (!noSorting || nClasses);

  try {
    // Extract the data from the table, compute the centers and fill the sortIndices
    EITERATE(ei, etable) {
      if (pushSortIndices)
        sortIndices.push_back(sortIndices.size());

      float *i_floatMap = new float[nColumns];
      tempFloatMap.push_back(i_floatMap);

      if (nClasses) {
        TValue &classVal = (*ei).getClass();
        const int tClass = classVal.isSpecial() ? nClasses : classVal.intV;
        classes.push_back(tClass);
        classBoundaries[tClass+1]++;
      }

      if (nColumns>1) {
        TExample::const_iterator eii((*ei).begin());

        float sumBri = 0.0;
        float sumBriX = 0.0;
        int sumX = 0;
        int N = 0;
        float thismax = -1e30f;
        float thismin = 1e30f;
      
        float *rai = i_floatMap;
        for(int xpoint = 0; xpoint<nColumns; rai++, eii++, xpoint++) {
          if ((*eii).isSpecial()) {
            *rai = UNKNOWN_F;
          }
          else {
            *rai = (*eii).floatV;
            sumBri += *rai;
            sumBriX += *rai * xpoint;
            sumX += xpoint;
            N += 1;
            if (*rai > thismax)
              thismax = *rai;
            if (*rai < thismin)
              thismin = *rai;
          }
        }

        tempLineAverages.push_back(N ? sumBri/N : UNKNOWN_F);
        tempLineCenters.push_back(N && (thismax != thismin) ? (sumBriX - thismin * sumX) / (sumBri - thismin * N) : UNKNOWN_F);
      }
      else {
        TValue val = *((*ei).begin());
        if (val.isSpecial()) {
          *i_floatMap = UNKNOWN_F;
          tempLineAverages.push_back(UNKNOWN_F);
          tempLineCenters.push_back(UNKNOWN_F);
        }
        else {
          *i_floatMap = val.floatV;
          tempLineAverages.push_back(val.floatV);
          tempLineCenters.push_back(val.floatV);
        }
      }
    }

    if (haveBase) {
      sortIndices = baseHeatmap->sortIndices;
      classBoundaries = baseHeatmap->classBoundaries;
    }

    else {
      if (nClasses)
        for(vector<int>::iterator cbi(classBoundaries.begin()+1), cbe(classBoundaries.end()); cbi!=cbe; *cbi += cbi[-1], cbi++);
    
      if (!noSorting) {
        if (nClasses) {
          CompareIndicesWClass compare(tempLineCenters, classes);
          sort(sortIndices.begin(), sortIndices.end(), compare);
        }
        else {
          CompareIndices compare(tempLineCenters);
          sort(sortIndices.begin(), sortIndices.end(), compare);
        }
      }
      else
        if (nClasses) {
          // stable sort by classes only
          vector<int> mcb(classBoundaries);
          int i = 0;
          sortIndices.resize(classes.size());
          for(vector<int>::const_iterator ci(classes.begin()), ce(classes.end()); ci!=ce; sortIndices[mcb[*(ci++)]++] = i++);
        }
    }

    floatMap.reserve(nRows);
    lineCenters.reserve(nRows);
    lineAverages.reserve(nRows);

	if (sortIndices.size()) {
      ITERATE(vector<int>, si, sortIndices) {
        esorted.addExample(etable[*si]);
        lineCenters.push_back(tempLineCenters[*si]);
        lineAverages.push_back(tempLineAverages[*si]);
        floatMap.push_back(tempFloatMap[*si]);
        tempFloatMap[*si] = NULL;
	  }
	}
	else {
	  sortedExamples = mlnew TExampleTable(PExampleGenerator(etable), false); // just references to examples, not copies
	  lineCenters = tempLineCenters;
	  lineAverages = tempLineAverages;
	  floatMap = tempFloatMap;
	}
  }
  catch (...) {
    ITERATE(vector<float *>, tfmi, tempFloatMap)
      delete *tfmi;
    ITERATE(vector<float *>, fmi, floatMap)
      delete *fmi;
    throw;
  }
}


THeatmapConstructor::~THeatmapConstructor()
{
  ITERATE(vector<float *>, fmi, floatMap)
    delete *fmi;
}


PHeatmapList THeatmapConstructor::operator ()(const float &unadjustedSqueeze, float &abslow, float &abshigh)
{
  abshigh = -1e30f;
  abslow = 1e30f;
    
  PHeatmapList hml = new THeatmapList;

  int *spec = new int[nColumns];
  
  for(int classNo = 0, ncl = nClasses ? nClasses : 1; classNo < ncl; classNo++) {
    const int classBegin = classBoundaries[classNo];
    const int classEnd = classBoundaries[classNo+1];

    if (classBegin == classEnd) {
      THeatmap *hm = new THeatmap(0, nColumns, sortedExamples);
      hml->push_back(hm);
      hm->exampleIndices->push_back(classBegin);
      hm->exampleIndices->push_back(classBegin);
      continue;
    }

    int nLines = int(floor(0.5 + (classEnd - classBegin) * unadjustedSqueeze));
    if (!nLines)
      nLines++;
    const float squeeze = float(nLines) / (classEnd-classBegin);

    THeatmap *hm = new THeatmap(nLines, nColumns, sortedExamples);
    hml->push_back(hm);

    float *fmi = hm->cells;
    float *ami = hm->averages;

    float inThisRow = 0;
    float *ri, *fri;
    int *si;
    int xpoint;
    vector<float>::const_iterator lavi(lineAverages.begin());

    int exampleIndex = classBegin;
    hm->exampleIndices->push_back(exampleIndex);

    for(vector<float *>::iterator rowi = floatMap.begin()+classBegin, rowe = floatMap.begin()+classEnd; rowi!=rowe; nLines--, inThisRow-=1.0, ami++, fmi+=nColumns) {
      for(xpoint = nColumns, ri = fmi, si = spec; xpoint--; *(ri++) = 0.0, *(si++) = 0);
      *ami = 0.0;
      int nDefinedAverages = 0;

      for(; (rowi != rowe) && ((inThisRow < 1.0) || (nLines==1)); inThisRow += squeeze, rowi++, exampleIndex++, lavi++) {
        for(xpoint = nColumns, fri = *rowi, ri = fmi, si = spec; xpoint--; fri++, ri++, si++)
          if (*fri != UNKNOWN_F) {
            *ri += *fri;
            (*si)++;
          }

        if (*lavi != UNKNOWN_F) {
          *ami += *lavi;
          nDefinedAverages++;
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

      *ami = nDefinedAverages ? *ami/nDefinedAverages : UNKNOWN_F;
    }
  }

  delete spec;

  return hml;
}


unsigned char *THeatmapConstructor::getLegend(const int &width, const int &height, const float &gamma, long &size) const
{
  float *fmp = new float[width], *fmpi = fmp;

  float wi1 = width-1;
  for(int wi = 0; wi<width; *(fmpi++) = (wi++)/wi1);
  
  unsigned char *legend = bitmap2string(1, height, 0, 1, size, fmp, width, 1, 0, 1, gamma, false);
  delete fmp;
  return legend;
}

