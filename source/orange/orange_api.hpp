#ifndef ORANGE_API

#ifdef _MSC_VER
  #pragma warning (disable : 4660 4661 4786 4114 4018 4267 4244 4702 4710 4290)
#endif

#ifdef _MSC_VER
  #ifdef ORANGE_EXPORTS
    #define ORANGE_API __declspec(dllexport)
    #define EXPIMP_TEMPLATE
  #else
    #define ORANGE_API __declspec(dllimport)
    #define EXPIMP_TEMPLATE
    #ifdef _DEBUG
      #pragma comment(lib, "orange_d.lib")
    #else
      #pragma comment(lib, "orange.lib")
    #endif
  #endif
#else
  #define ORANGE_API
  #define EXPIMP_TEMPLATE
#endif

#endif
