#ifndef __RCONVERSIONS_HPP
#define __RCONVERSIONS_HPP


#include "root.hpp"

WRAPPER(ExampleGenerator)
void exampleGenerator2r(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                          double *&X, double *&y, double *&w, int &rows, int &columns);

void parseMatrixContents(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                         bool &hasClass, bool &classVector, bool &multiClassVector, bool &weightVector, bool &classIsDiscrete, int &columns,
                         vector<bool> &include);

#endif
