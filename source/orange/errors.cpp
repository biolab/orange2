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


#include "errors.hpp"
#include "stdio.h"

// We include Python because it sets up everything for va_start
#include "Python.h"

bool exhaustiveWarnings = false;

char excbuf[512], excbuf2[512];

void raiseError(const char *anerr, ...)
{ va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, anerr);
  #else
    va_start(vargs);
  #endif

  vsnprintf(excbuf, 512, anerr, vargs);
  throw mlexception(excbuf);
}


void raiseErrorWho(const char *who, const char *anerr, ...)
{ va_list vargs;
  #ifdef HAVE_STDARG_PROTOTYPES
    va_start(vargs, anerr);
  #else
    va_start(vargs);
  #endif

  snprintf(excbuf2, 512, "%s: %s", who, anerr);
  vsnprintf(excbuf, 512, excbuf2, vargs);
  throw mlexception(excbuf);
}

