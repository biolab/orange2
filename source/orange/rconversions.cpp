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

#include <vector>
#include "rconversions.hpp"
#include "vars.hpp"
#include "examplegen.hpp"


void exampleGenerator2r(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                        double *&X, double *&y, double *&w, int &rows, int &columns)
{
  bool hasClass, classVector, multiclassVector, weightVector, classIsDiscrete;
  vector<bool> include;

  parseMatrixContents(egen, weightID, contents, multiTreatment,
                          hasClass, classVector, multiclassVector, weightVector, classIsDiscrete, columns, include);

  // this does not work if the domain includes multiple classes -- which is irrelevant
  // since nobody calls this function anyway ;)

  rows = egen->numberOfExamples();

  X = columns ? (double *)malloc(rows * columns * sizeof(double)) : NULL;
  y = classVector ? (double *)malloc(rows * sizeof(double)) : NULL;
  w = weightVector ? (double *)malloc(rows * sizeof(double)) : NULL;

  double *Xi = X;
  double *yi = y;
  double *wi = w;

  try {
    int row = 0;
    TExampleGenerator::iterator ei(egen->begin());
    for(; ei; ++ei, row++, Xi += 1 - rows*columns) {
      int col = 0;
      
      /* This is all optimized assuming that each symbol (A, C, W) only appears once. 
         If it would be common for them to appear more times, we could cache the values,
         but since this is unlikely, caching would usually slow down the conversion */
      for(const char *cp = contents; *cp && (*cp!='/'); cp++) {
        switch (*cp) {
          case 'A':
          case 'a': {
            const TVarList &attributes = egen->domain->attributes.getReference();
            TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
            TExample::iterator eei((*ei).begin());
            vector<bool>::const_iterator bi(include.begin());
            for(; vi != ve; eei++, vi++, bi++)
              if (*bi) {
                if ((*eei).isSpecial())
                  raiseErrorWho("exampleGenerator2r", "value of attribute '%s' in example '%i' is undefined", (*vi)->get_name().c_str(), row);
                *Xi = (*vi)->varType == TValue::FLOATVAR ? (*eei).floatV : float((*eei).intV);
                Xi += rows;
              }
            break;
          }

          case 'C':
          case 'c': 
            if (hasClass) {
              const TValue &classVal = (*ei).getClass();
              if (classVal.isSpecial())
                raiseErrorWho("exampleGenerator2r", "example %i has undefined class", row);
              *Xi = classIsDiscrete ? float(classVal.intV) : classVal.floatV;
              Xi += rows;
            }
            break;

          case 'W':
          case 'w': 
            if (weightID)
              *Xi = WEIGHT(*ei);
              Xi += rows;
            break;

          case '0':
            *Xi = 0.0;
            Xi += rows;
            break;

          case '1':
            *Xi = 1.0;
            Xi += rows;
            break;
        }
      }

      if (y) {
        const TValue &classVal = (*ei).getClass();
        if (classVal.isSpecial())
          raiseErrorWho("exampleGenerator2r", "example %i has undefined class", row);
        *(yi++) = classIsDiscrete ? float(classVal.intV) : classVal.floatV;
      }

      if (w)
        *(wi++) = WEIGHT(*ei);
    }
  }
  catch (...) {
    if (X)
      free(X);
    if (y)
      free(y);
    if (w)
      free(w);
    throw;
  }
}
