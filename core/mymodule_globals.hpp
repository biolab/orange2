#ifndef __MYMODULE_GLOBALS
#define __MYMODULE_GLOBALS

// The usual export-import mechanism; MYMODULE_EXPORTS is defined by default
//  (VC takes care of this)
#ifdef _MSC_VER
    #ifdef MYMODULE_EXPORTS
        #define MYMODULE_API __declspec(dllexport)
    #else
        #define MYMODULE_API __declspec(dllimport)
    
        #ifdef _DEBUG
            #pragma comment(lib, "mymodule_d.lib")
        #else
            #pragma comment(lib, "mymodule.lib")
        #endif
    #endif
#else
    #define MYMODULE_API
#endif

// garbage collection stuff
#include "garbage.hpp"
#define MMWRAPPER(x) BASIC_WRAPPER(x, MYMODULE_API)
#define MMVWRAPPER(x) BASIC_VWRAPPER(x, MYMODULE_API)


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