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

#ifdef _MSC_VER
#define mlexception exception
#else
class mlexception : public exception {
public:
   string err_desc;
   mlexception(const string &des);
   ~mlexception() throw();
   virtual const char* what () const throw();
};
#endif


void raiseError(const char *anerr, ...);
void raiseErrorWho(const char *who, const char *anerr, ...);


exception TUserError(const string &anerr);
exception TUserError(const string &anerr, const string &s);
exception TUserError(const string &anerr, const string &s1, const string &s2);
exception TUserError(const string &anerr, const string &s1, const string &s2, const string &s3);
exception TUserError(const string &anerr, const string &s1, const string &s2, const string &s3, const string &s4);
exception TUserError(const string &anerr, const long i);

exception TUserError(const char *anerr);
exception TUserError(const char *anerr, const char *s);
exception TUserError(const char *anerr, const char *s1, const char *s2);
exception TUserError(const char *anerr, const char *s1, const char *s2, const char *s3);
exception TUserError(const char *anerr, const char *s1, const char *s2, const char *s3, const char *s4);
exception TUserError(const char *anerr, const long i);

#define TRY try {
#define CATCH } catch(exception err) { fprintf(stderr, "Error: %s\n", err.what()); }


#define _QUOTE(x) # x
#define QUOTE(x) _QUOTE(x)
#define __FILE__LINE__ __FILE__ "(" QUOTE(__LINE__) ") : "

#define NOTE( x )  message( x )
#define FILE_LINE  message( __FILE__LINE__ )

#define TODO( x )  message( __FILE__LINE__"\n"           \
        " ------------------------------------------------\n" \
        "|  TODO :   " x "\n" \
        " -------------------------------------------------\n" )
#define FIXME( x )  message(  __FILE__LINE__"\n"           \
        " ------------------------------------------------\n" \
        "|  FIXME :  " x "\n" \
        " -------------------------------------------------\n" )
#define todo( x )  message( __FILE__LINE__" TODO :   " x "\n" ) 
#define fixme( x )  message( __FILE__LINE__" FIXME:   " x "\n" ) 

#endif

