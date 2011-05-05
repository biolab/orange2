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


#ifndef __GSLCONVERSIONS_HPP
#define __GSLCONVERSIONS_HPP


#define GSL_FREE(O,tpe) { if (O) { gsl_##tpe##_free(O); O = NULL; }}

#include "gsl_imports.hpp"

#include "root.hpp"

WRAPPER(ExampleGenerator)
void exampleGenerator2gsl(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                          void *&X, void *&y, void *&w, int &rows, int &columns);

void parseMatrixContents(PExampleGenerator egen, const int &weightID, const char *contents, const int &multiTreatment,
                         bool &hasClass, bool &classVector, bool &weightVector, bool &classIsDiscrete, int &columns,
                         vector<bool> &include);

#endif
