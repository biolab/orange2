/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


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
