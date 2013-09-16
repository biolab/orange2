#ifndef __GSLCONVERSIONS_HPP
#define __GSLCONVERSIONS_HPP


#define GSL_FREE(O,tpe) { if (O) { gsl_##tpe##_free(O); O = NULL; }}

#include "gsl_imports.hpp"

#include "root.hpp"

WRAPPER(ExampleGenerator)
void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                          void *&X, void *&y, void *&w, int &rows, int &columns);

void parseMatrixContents(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                         bool &hasClass, bool &classVector, bool &multiclassVector, bool &weightVector, bool &classIsDiscrete, int &columns,
                         vector<bool> &include);

#endif
