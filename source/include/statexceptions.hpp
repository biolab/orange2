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
