#if !defined(GENERAL_H)
#define GENERAL_H

//#define CSET
//#define BORLAND
#define MICROSOFT
//#define UNIX

//#define DEBUG
//#define DEBUG_NEW
//#include "new_new.h"

#define RAMP_FUNCTION

#if defined(UNIX)
  const char DirSeparator = '/' ;
  #define strDirSeparator  "/"
#else
  const char DirSeparator = '\\' ;
  #define strDirSeparator  "\\"
#endif

const int MaxPath = 1024 ;
const int MaxNameLen = 1024;
const int MaxFileNameLen = 512 ;
const int MaxIntLen = 32 ;
// const char MaxChar = '\xff' ;
enum boolean { FALSE=0, TRUE=1 } ;
typedef char* Pchar ;
typedef int* Pint ;
typedef float* Pfloat ;
typedef double* Pdouble ;
enum attributeCount  {aDISCRETE, aCONTINUOUS} ;


#if defined(MICROSOFT)
#pragma warning (disable:4146 4056)
#endif

#endif

