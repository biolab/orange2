#ifndef __GSLCONVERSIONS_HPP
#define __GSLCONVERSIONS_HPP

#define GSL_FREE(O,tpe) { if (O) { gsl_##tpe##_free(O); O = NULL; }}

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

#include "root.hpp"

WRAPPER(ExampleGenerator)
void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, bool addConstant, const int &multiTreatment,
                          gsl_matrix *&X, gsl_vector *&y, gsl_vector *&w, int &rows, int &columns);

#endif