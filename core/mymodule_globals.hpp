#ifndef __COREMODULE_GLOBALS
#define __COREMODULE_GLOBALS

#ifdef _MSC_VER
    #ifdef CORE_EXPORTS
        #define CORE_API __declspec(dllexport)
    #else
        #define CORE_API __declspec(dllimport)
    
        #ifdef _DEBUG
            #pragma comment(lib, "core_d.lib")
        #else
            #pragma comment(lib, "core.lib")
        #endif
    #endif
#else
    #define CORE_API
#endif

// garbage collection stuff
#include "garbage.hpp"
#define COREWRAPPER(x) BASIC_WRAPPER(x, CORE_API)
#define COREVWRAPPER(x) BASIC_VWRAPPER(x, CORE_API)


// wrapper macro for orange classes is defined here (besides other things)
#include "orange_api.hpp"

// we need this for things like cc_Variable
#include "../orange/px/externs.px"

// these are some wrapping macros for our classes
#include "externs.px"

// mostly dummy macros like C_NAMED, C_CALL, PYARGS ...
#include "../pyxtract/pyxtract_macros.hpp"

#define PyCATCH_r(e) PyCATCH_r_et(e,PyExc_OrangeKernel)

#endif