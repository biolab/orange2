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

