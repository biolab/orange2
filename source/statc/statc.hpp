#ifdef _MSC_VER
  #ifdef STATC_EXPORTS
    #define STATC_API __declspec(dllexport)
  #else
    #define STATC_API __declspec(dllimport)
  #endif
#else
  #define STATC_API
#endif


#include "c2py.hpp"

extern "C" STATC_API void initstatc(void);
extern PyMethodDef statc_functions[];

