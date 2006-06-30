#ifndef __ORANGENE_GLOBALS
#define __ORANGENE_GLOBALS

#include "garbage.hpp"

#ifdef _MSC_VER
    #ifdef ORANGENE_EXPORTS
        #define ORANGENE_API __declspec(dllexport)
        #define EXPIMP_TEMPLATE
    #else
        #define ORANGENE_API __declspec(dllimport)
        #define EXPIMP_TEMPLATE extern
    
        #ifdef _DEBUG
            #pragma comment(lib, "orangene_d.lib")
        #else
            #pragma comment(lib, "orangene.lib")
        #endif
    #endif
#else
    #define ORANGENE_API
    #define EXPIMP_TEMPLATE
#endif

#define OGWRAPPER(x) BASIC_WRAPPER(x, ORANGENE_API)
#define OGVWRAPPER(x) BASIC_VWRAPPER(x, ORANGENE_API)


#include "../pyxtract/pyxtract_macros.hpp"

#define PyCATCH_r(e) PyCATCH_r_et(e,PyExc_OrangeKernel)

#endif
