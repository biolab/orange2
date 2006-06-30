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


#elif defined LINUX || defined FREEBSD

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

#elif defined DARWIN


THIS CODE NEEDS TO BE REWRITTEN (it is copied directly from C45 loader)

#include <mach-o/dyld.h>
#include <string.h>

void *getsym(NSModule &newModule, const char *name)
{
  NSSymbol theSym = NSLookupSymbolInModule(newModule, name);
  if (!theSym)
    raiseErrorWho("C45Loader", "invalid %s, cannot find symbol %s", C45NAME, name);
  return theSym;
}

void dynloadC45(char pathname[])
{
  NSObjectFileImageReturnCode rc;
  NSObjectFileImage image;
  NSModule newModule;
  
  rc = NSCreateObjectFileImageFromFile(pathname, &image);
  if (rc != NSObjectFileImageSuccess)
    raiseErrorWho("C45Loader", "Cannot load %s", C45NAME);
      
  newModule = NSLinkModule(image, pathname, NSLINKMODULE_OPTION_BINDNOW | \
                                            NSLINKMODULE_OPTION_RETURN_ON_ERROR | \
                                            NSLINKMODULE_OPTION_PRIVATE); 
  if (!newModule)
    raiseErrorWho("C45Loader", "Cannot link module %s", C45NAME);

  pc45data = getsym(newModule, "_c45Data");
  c45learn = (learnFunc *)getsym(newModule, "_learn");
  c45garbage = (garbageFunc *)getsym(newModule, "_guarded_collect");
}
    
#else

void dynloadC45(char [])
{ raiseErrorWho("C45Loader", "c45 is not supported on this platform"); }

#endif
