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


#ifdef _MSC_VER
  #define NOMINMAX
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
  #include <windows.h>
#endif

#include "Python.h"
#include "orangeom_globals.hpp"

PyObject *py_triangulate(PyObject *, PyObject *args, PyObject *);


PyMethodDef orangeom_functions[] = {
     {"triangulate", (binaryfunc)py_triangulate, METH_VARARGS},
     {NULL, NULL}
};


extern "C" ORANGEOM_API void initorangeom()
{ Py_InitModule("orangeom", orangeom_functions); }


#ifdef _MSC_VER
BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{ switch (ul_reason_for_call)
	{ case DLL_PROCESS_ATTACH:case DLL_THREAD_ATTACH:case DLL_THREAD_DETACH:case DLL_PROCESS_DETACH:break; }
  return TRUE;
}
#endif
