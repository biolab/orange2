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
