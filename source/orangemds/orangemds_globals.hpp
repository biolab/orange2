
#ifndef __ORANGEMDS_GLOBALS
#define __ORANGEMDS_GLOBALS

#include "orange_api.hpp"
#include "garbage.hpp"

#ifdef _MSC_VER
  #ifdef ORANGEMDS_EXPORTS
    #define ORANGEMDS_API __declspec(dllexport)
  #else
    #define ORANGEMDS_API __declspec(dllimport)
  #endif
#else
  #define ORANGEMDS_API
#endif

#define OMWRAPPER(x) BASIC_WRAPPER(x, ORANGEMDS_API)
#define OMVWRAPPER(x) BASIC_VWRAPPER(x, ORANGEMDS_API)

#include "../pyxtract/pyxtract_macros.hpp"

#endif

