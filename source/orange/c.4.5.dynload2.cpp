#if defined _MSC_VER

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#include <windows.h>

typedef void *learnFunc(char gainRatio, char subset, char batch, char probThresh,
                       int trials, int minObjs, int window, int increment, float cf, char prune);
typedef void garbageFunc();

extern learnFunc *c45learn;
extern garbageFunc *c45garbage;
extern void *pc45data;

const char *dynloadC45(char buf[], char *bp)
{
  #ifdef _DEBUG
  strcpy(bp, "\\c45_d.dll");
  #else
  strcpy(bp, "\\c45.dll");
  #endif

  HINSTANCE c45Dll = LoadLibrary(buf);
  if (!c45Dll)
    return "cannot load c45.dll";

//  char funcname[258];
  
  //snprintf(funcname, sizeof(funcname), "%s", "c45Data");
  pc45data = GetProcAddress(c45Dll, "c45Data");

  //snprintf(funcname, sizeof(funcname), "%s", "learn");
  c45learn = (learnFunc *)(GetProcAddress(c45Dll, "learn"));

  //snprintf(funcname, sizeof(funcname), "%s", "guarded_collect");
  c45garbage = (garbageFunc *)(GetProcAddress(c45Dll, "guarded_collect"));
 
  if (!pc45data || !c45learn || !c45garbage)
    return "c45.dll is invalid";

  return NULL;
}

#elif defined LINUX

#include <dlfcn.h>

const char *dynloadC45(char buf[], char *bp)
{ 
  #ifdef _DEBUG
  strcpy(bp, "/c45_d.so");
  #else
  strcpy(bp, "/c45.so");
  #endif

  void *handle = dlopen(buf, 0 /*dlopenflags*/);
  if (handle == NULL)
    return dlerror();
  
//  getDataFunc *p = (getDataFunc *) dlsym(handle, "_Z10getc45Datav");
  pc45data = dlsym(handle, "c45Data");
  c45learn = (learnFunc *) dlsym(handle, "learn");
  c45garbage = (garbageFunc *) dlsym(handle, "guarded_collect");
  
  if (!pc45data || !c45learn || !c45garbage)
    return "c45.so is invalid (required functions are not found)";

  return NULL;
}

#elif defined DARWIN

#else

const char *dynloadC45()
{ return "C45Loader", "c45 is not supported on this platform"; }

#endif

