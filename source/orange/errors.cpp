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

