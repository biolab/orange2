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


#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

#include <exception>
#include <string>

using namespace std;

#ifdef _MSC_VER

#define statexception exception

#else

class statexception : public exception {
public:
   string err_desc;
   statexception(const string &des);
   ~statexception() throw();
   virtual const char* what () const throw();
};


#endif


exception StatException(const string &anerr);
exception StatException(const string &anerr, const string &s);
exception StatException(const string &anerr, const string &s1, const string &s2);
exception StatException(const string &anerr, const string &s1, const string &s2, const string &s3);
exception StatException(const string &anerr, const long i);


#endif
