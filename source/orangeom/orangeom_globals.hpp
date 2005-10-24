#ifndef __ORANGEOM_GLOBALS
#define __ORANGEOM_GLOBALS

#include "garbage.hpp"

#ifdef _MSC_VER
    #ifdef ORANGEOM_EXPORTS
        #define ORANGEOM_API __declspec(dllexport)
        #define EXPIMP_TEMPLATE
    #else
        #define ORANGEOM_API __declspec(dllimport)
        #define EXPIMP_TEMPLATE extern
    
        #ifdef _DEBUG
            #pragma comment(lib, "orange_d.lib")
        #else
            #pragma comment(lib, "orange.lib")
        #endif
    #endif
#else
    #define ORANGEOM_API
    #define EXPIMP_TEMPLATE
#endif

#define OMWRAPPER(x) BASIC_WRAPPER(x, ORANGEOM_API)
#define OMVWRAPPER(x) BASIC_VWRAPPER(x, ORANGEOM_API)


#include "../pyxtract/pyxtract_macros.hpp"

#endif
