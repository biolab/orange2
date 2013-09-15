#ifndef __ERRORS_HPP
#define __ERRORS_HPP

#include <string>
#include "px/orange_globals.hpp"

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

void ORANGE_API raiseError(const char *anerr, ...);
void ORANGE_API raiseErrorWho(const char *who, const char *anerr, ...);

#endif

