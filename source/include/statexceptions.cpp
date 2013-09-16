#ifdef _MSC_VER
  #pragma warning (disable : 4290)
#endif

#include <stdio.h>
#include "statexceptions.hpp"

#ifndef _MSC_VER
statexception::statexception(const string &desc)
  : err_desc(desc)
  {}

const char* statexception::what () const throw()
  { return err_desc.c_str(); };

statexception::~statexception() throw()
{}
#endif



exception StatException(const string &anerr)
{ return statexception(anerr.c_str()); }

exception StatException(const string &anerr, const string &s)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s.c_str());
  return statexception(buf);
}

exception StatException(const string &anerr, const string &s1, const string &s2)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str());
  return statexception(buf);
}

exception StatException(const string &anerr, const string &s1, const string &s2, const string &s3)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str(), s3.c_str());
  return statexception(buf);
}

exception StatException(const string &anerr, const long i)
{ char buf[255];
  sprintf(buf, anerr.c_str(), i);
  return statexception(buf);
}

