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

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include <vector>

#include "vars.hpp"
#include "examplegen.hpp"


#ifndef _DEBUG
// I prefer to see exceptions when debugging
extern "C" void my_gsl_error_handler(const char *reason, const char *file, int line, int)
{ raiseErrorWho("GSL", "%s (%s:%i)", reason, file, line); }

gsl_error_handler_t *fooerrh = gsl_set_error_handler(my_gsl_error_handler);
#endif


#define convertToFloat(val, name) \
 if (val.isSpecial()) \
    raiseError("exampleGenerator2gsl: value of attribute '%s' in example '%i' is undefined", name, row); \
 flt = val.varType == TValue::INTVAR ? float(val.intV) : val.floatV; \

void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, bool addConstant, const int &multiTreatment, gsl_matrix *&X, gsl_vector *&y, gsl_vector *&w, int &rows, int &columns)
{
  if (!egen->domain->classVar)
    raiseErrorWho("exampleGenerator2gsl", "classless domain");

  const TVariable &classVar = egen->domain->classVar.getReference();
  TEnumVariable *eclassVar = egen->domain->classVar.AS(TEnumVariable);
  const TVarList &attributes = egen->domain->attributes.getReference();

  if (eclassVar ? (eclassVar->values->size()>2) && (multiTreatment != 1)
                : classVar.varType != TValue::FLOATVAR)
    raiseErrorWho("exampleGenerator2gsl", "class should be continuous or binary, unless explicitly treated as ordinal");

  rows = egen->numberOfExamples();
  columns = addConstant ? 1 : 0;

  vector<bool> include;
  const_ITERATE(TVarList, vi, attributes) {
    if ((*vi)->varType == TValue::FLOATVAR) {
      columns++;
      include.push_back(true);
    }
    else if ((*vi)->varType == TValue::INTVAR) {
      if ((*vi).AS(TEnumVariable)->values->size() == 2) {
        columns++;
        include.push_back(true);
      }
      else
        switch (multiTreatment) {
          case 0:
            include.push_back(false);
            break;

          case 1:
            columns++;
            include.push_back(true);
            break;

          default:
            raiseErrorWho("exampleGenerator2gsl: attribute '%s' is multinomial", (*vi)->name.c_str());
        }
    }
    else
      raiseErrorWho("exampleGenerator2gsl: attribute '%s' is of unsupported type", (*vi)->name.c_str());
  }

  y = gsl_vector_alloc(rows);
  w = gsl_vector_alloc(rows);
  X = gsl_matrix_alloc(rows, columns);

  try {
    int row = 0;
    TExampleGenerator::iterator ei(egen->begin());
    for(; ei; ++ei, row++) {
      int col = 0;
      if (addConstant)
        gsl_matrix_set(X, row, col++, 1.0);

      TVarList::const_iterator vi(attributes.begin()), ve(attributes.end());
      TExample::iterator eei((*ei).begin());
      vector<bool>::const_iterator bi(include.begin());
      for(; vi != ve; eei++, vi++, bi++)
        if (*bi) {
          if ((*eei).isSpecial())
            raiseError("exampleGenerator2gsl: value of attribute '%s' in example '%i' is undefined", (*vi)->name.c_str(), row);
          gsl_matrix_set(X, row, col++, (*vi)->varType == TValue::FLOATVAR ? (*eei).floatV : float((*eei).intV));
        }

      const TValue &classVal = (*ei).getClass();
      if (classVal.isSpecial())
        raiseError("exampleGenerator2gsl: example %i has undefined class", row);
      gsl_vector_set(y, row, classVal.varType == TValue::FLOATVAR ? classVal.floatV : float(classVal.intV));

      gsl_vector_set(w, row, WEIGHT(*ei));
    }
  }
  catch (...) {
    gsl_vector_free(y);
    gsl_vector_free(w);
    gsl_matrix_free(X);
    throw;
  }
}