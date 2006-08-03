#include "jit_linker.hpp"

#if defined _MSC_VER

#include <direct.h>
#define getcwd _getcwd

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#include <windows.h>

int jit_link(const char *dllname, TJitLink *functions, TDefaultFunc deffunc)
{
  TJitLink *function = functions;

  HINSTANCE jitDll = LoadLibrary(dllname);
  if (jitDll) {
    for(; function->address; function++) {
      void *sym = GetProcAddress(jitDll, function->funcname);
      if (!sym)
        break;
      *function->address = sym;
    }
  }

	if (function->address) {
	  for(function = functions; function->address; function++)
	    *function->address = deffunc;

    if (jitDll) {
	    FreeLibrary(jitDll);
      return -1;
    }
    else
      return -2;
	}

  return 0;
}


#elif defined LINUX || defined FREEBSD || defined DARWIN

#include <dlfcn.h>
#include <unistd.h>

int jit_link(const char *dllname, TJitLink *functions, TDefaultFunc deffunc)
{
  TJitLink *function = functions;

  void *jitDll = dlopen(dllname, 0);
  if (jitDll) {
    for(; function->address; function++) {
      void *sym = dlsym(jitDll, function->funcname);
      if (!sym)
        break;
      *function->address = sym;
    }
  }

	if (function->address) {
	  for(function = functions; function->address; function++)
	    *function->address = deffunc;

    if (jitDll) {
	    dlclose(jitDll);
      return -1;
    }
    else
      return -2;
	}

  return 0;
}
   
#else

void dynloadC45(char [])
{ raiseErrorWho("C45Loader", "c45 is not supported on this platform"); }

#endif
