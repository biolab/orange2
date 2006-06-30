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

/*
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
*/
#include <vector>

#include "vars.hpp"
#include "examplegen.hpp"

/*
#ifndef _DEBUG
// I prefer to see exceptions when debugging
extern "C" void my_gsl_error_handler(const char *reason, const char *file, int line, int)
{ raiseErrorWho("GSL", "%s (%s:%i)", reason, file, line); }

gsl_error_handler_t *fooerrh = gsl_set_error_handler(my_gsl_error_handler);
#endif
*/

void parseMatrixContents(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                         bool &hasClass, bool &classVector, bool &weightVector, bool &classIsDiscrete, int &columns,
                         vector<bool> &include)
{
  hasClass = bool(egen->domain->classVar);

  columns = 0;
  int classIncluded = 0, attrsIncluded = 0, weightIncluded = 0;
  bool attrsRequested = false, classRequested = false, weightRequested = false;
  const char *cp;
  for(cp = contents; *cp && (*cp!='/'); cp++) {
    switch (*cp) {
      case 'A': attrsRequested = true;
      case 'a': attrsIncluded++;
                break;

      case 'C': classRequested = true;
      case 'c': classIncluded++;
                break;

      case 'W': weightRequested = true;
      case 'w': weightIncluded++;
                break;

      case '0':
      case '1': columns++;
                break;
      default:
        raiseErrorWho("parseMatrixContents", "unrecognized character '%c' in format string '%s')", *cp, contents);
    }
  }

  classVector = false;
  weightVector = false;

  if (*cp)
    while(*++cp)
      switch (*cp) {
        case 'A':
        case 'a': raiseErrorWho("parseMatrixContents", "invalid format string (attributes on the right side)");

        case '0':
        case '1': raiseErrorWho("parseMatrixContents", "invalid format string (constants on the right side)");

        case 'c': classVector = hasClass; break;
        case 'C': classVector = true; break;

        case 'w': weightVector = (weightID != 0); break;
        case 'W': weightVector = true; break;
        default:
          raiseErrorWho("parseMatrixContents", "unrecognized character '%c' in format string '%s')", *cp, contents);
      }


  if (classIncluded || classVector) {
    if (hasClass) {
      TEnumVariable *eclassVar = egen->domain->classVar.AS(TEnumVariable);
      classIsDiscrete = eclassVar != NULL;
      if (classIsDiscrete) {
        if ((eclassVar->values->size()>2) && (multiTreatment != 1))
          raiseErrorWho("parseMatrixContents", "multinomial classes are allowed only when explicitly treated as ordinal");  
      }
      else {
        if (egen->domain->classVar->varType != TValue::FLOATVAR)
          raiseErrorWho("parseMatrixContents", "unsupported class type");
      }  

      columns += classIncluded;
    }
    else if (classRequested || classVector)
      raiseErrorWho("parseMatrixContents", "classless domain");
  }


  if (weightIncluded || weightVector) {
    if (weightID)
      columns += weightIncluded;
  }

  include.clear();

  if (attrsIncluded) {
    int attrs_in = 0;

    const_PITERATE(TVarList, vi, egen->domain->attributes) {
      if ((*vi)->varType == TValue::FLOATVAR) {
        attrs_in++;
        include.push_back(true);
      }
      else if ((*vi)->varType == TValue::INTVAR) {
        if ((*vi).AS(TEnumVariable)->values->size() == 2) {
          attrs_in++;
          include.push_back(true);
        }
        else
          switch (multiTreatment) {
            case 0:
              include.push_back(false);
              break;

            case 1:
              attrs_in++;
              include.push_back(true);
              break;

            default:
              raiseErrorWho("parseMatrixContents", "attribute '%s' is multinomial", (*vi)->name.c_str());
          }
      }
      else
        raiseErrorWho("parseMatrixContents", "attribute '%s' is of unsupported type", (*vi)->name.c_str());
    }

    if (attrsRequested && !attrs_in)
      raiseErrorWho("parseMatrixContents", "the domain has no (useful) attributes");

    columns += attrs_in * attrsIncluded;
  }
}


/*
void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                          gsl_matrix *&X, gsl_vector *&y, gsl_vector *&w, int &rows, int &columns)
{
  bool hasClass, classVector, weightVector, classIsDiscrete;
  vector<bool> include;

  parseMatrixContents(egen, weightID, contents, multiTreatment,
                          hasClass, classVector, weightVector, classIsDiscrete, columns, include);

  rows = egen->numberOfExamples();

  X = columns ? gsl_matrix_calloc(rows, columns) : NULL;
  y = classVector ? gsl_vector_calloc(rows) : NULL;
  w = weightVector ? gsl_vector_calloc(rows) : NULL;

  try {
    int row = 0;
    TExampleGenerator::iterator ei(egen->begin());
    for(; ei; ++ei, row++) {
      int col = 0;
      
      /* This is all optimized assuming that each symbol (A, C, W) only appears once. 
         If it would be common for them, we could cache the values, but since this is
         unlikely, caching would only slow down the conversion 
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
                  raiseErrorWho("exampleGenerator2gsl", "value of attribute '%s' in example '%i' is undefined", (*vi)->name.c_str(), row);
                gsl_matrix_set(X, row, col++, (*vi)->varType == TValue::FLOATVAR ? (*eei).floatV : float((*eei).intV));
              }
            break;
          }

          case 'C':
          case 'c': 
            if (hasClass) {
              const TValue &classVal = (*ei).getClass();
              if (classVal.isSpecial())
                raiseErrorWho("exampleGenerator2gsl", "example %i has undefined class", row);
              gsl_matrix_set(X, row, col++, classIsDiscrete ? float(classVal.intV) : classVal.floatV);
            }
            break;

          case 'W':
            gsl_matrix_set(X, row, col++, weightID ? WEIGHT(*ei) : 1.0);
            break;

          case 'w': 
            if (weightID)
              gsl_matrix_set(X, row, col++, WEIGHT(*ei));
            break;

          case '0':
            gsl_matrix_set(X, row, col++, 0.0);
            break;

          case '1':
            gsl_matrix_set(X, row, col++, 1.0);
            break;
        }
      }

      if (y) {
        const TValue &classVal = (*ei).getClass();
        if (classVal.isSpecial())
          raiseErrorWho("exampleGenerator2gsl", "example %i has undefined class", row);
        gsl_vector_set(y, row, classVal.varType == TValue::FLOATVAR ? classVal.floatV : float(classVal.intV));
      }

      if (w)
        gsl_vector_set(w, row, weightID ? WEIGHT(*ei) : 1.0);
    }
  }
  catch (...) {
    if (X)
      gsl_matrix_free(X);
    if (y)
      gsl_vector_free(y);
    if (w)
      gsl_vector_free(w);
    throw;
  }
}

*/
