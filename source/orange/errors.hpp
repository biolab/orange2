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

#ifndef __ERRORS_HPP
#define __ERRORS_HPP

#include <string>

using namespace std;

extern bool exhaustiveWarnings;

#ifdef _MSC_VER
#define mlexception exception
#else
class mlexception : public exception {
public:
   string err_desc;

   mlexception(const string &desc)
   : err_desc(desc)
   {}

   ~mlexception() throw()
   {}

   virtual const char* what () const throw()
   { return err_desc.c_str(); };
};
#endif

#include "garbage.hpp"

void ORANGE_API raiseError(const char *anerr, ...);
void ORANGE_API raiseErrorWho(const char *who, const char *anerr, ...);

#endif

