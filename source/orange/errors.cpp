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


#include "garbage.hpp"
#include "errors.hpp"

#ifndef _MSC_VER
mlexception::mlexception(const string &desc)
  : err_desc(desc)
  {}

mlexception::~mlexception() throw()
{}

const char* mlexception::what () const throw()
  { return err_desc.c_str(); };
#endif

exception TUserError(const string &anerr)
{ return mlexception(anerr.c_str()); }


exception TUserError(const string &anerr, const string &s)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s.c_str());
  return mlexception(buf);
}


exception TUserError(const string &anerr, const string &s1, const string &s2)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str());
  return mlexception(buf);
}


exception TUserError(const string &anerr, const string &s1, const string &s2, const string &s3)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str(), s3.c_str());
  return mlexception(buf);
}

exception TUserError(const string &anerr, const string &s1, const string &s2, const string &s3, const string &s4)
{ char buf[255];
  sprintf(buf, anerr.c_str(), s1.c_str(), s2.c_str(), s3.c_str(), s4.c_str());
  return mlexception(buf);
}


exception TUserError(const string &anerr, const long i)
{ char buf[255];
  sprintf(buf, anerr.c_str(), i);
  return mlexception(buf);
}




exception TUserError(const char *anerr)
{ return mlexception(anerr); }


exception TUserError(const char *anerr, const char *s)
{ char buf[255];
  sprintf(buf, anerr, s);
  return mlexception(buf);
}


exception TUserError(const char *anerr, const char *s1, const char *s2)
{ char buf[255];
  sprintf(buf, anerr, s1, s2);
  return mlexception(buf);
}


exception TUserError(const char *anerr, const char *s1, const char *s2, const char *s3)
{ char buf[255];
  sprintf(buf, anerr, s1, s2, s3);
  return mlexception(buf);
}

exception TUserError(const char *anerr, const char *s1, const char *s2, const char *s3, const char *s4)
{ char buf[255];
  sprintf(buf, anerr, s1, s2, s3, s4);
  return mlexception(buf);
}


exception TUserError(const char *anerr, const long i)
{ char buf[255];
  sprintf(buf, anerr, i);
  return mlexception(buf);
}



char excbuf[512], excbuf2[256];

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

  snprintf(excbuf2, 256, "%s: %s", who, anerr);
  vsnprintf(excbuf, 512, excbuf2, vargs);
  throw mlexception(excbuf);
}

