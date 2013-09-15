#ifdef _MSC_VER
  #ifdef CORN_EXPORTS
    #define CORN_API __declspec(dllexport)
  #else
    #define CORN_API __declspec(dllimport)
  #endif
#else
  #define CORN_API
#endif


#include "c2py.hpp"

extern "C" CORN_API void initcorn(void);
extern PyMethodDef corn_functions[];

