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


#ifndef __ORANGE_HPP
#define __ORANGE_HPP

#include "pyxtract_macros.hpp"

#ifdef _MSC_VER
  #ifdef ORANGE_EXPORTS
    #define ORANGE_API __declspec(dllexport)
  #else
    #define ORANGE_API __declspec(dllimport)
  #endif
#else
  #define ORANGE_API
#endif


#undef min
#undef max
#include "c2py.hpp"
#include "root.hpp"

extern PyObject *orangeModule;

extern "C" ORANGE_API void initorange(void);


extern PyObject *PyExc_OrangeKernel, *PyExc_OrangeAttributeWarning, *PyExc_OrangeWarning;

#define PyTRY try {

#define PYNULL ((PyObject *)NULL)
#define PyCATCH   PyCATCH_r(PYNULL)
#define PyCATCH_1 PyCATCH_r(-1)

#define PyCATCH_r(r) \
  } \
catch (pyexception err)   { err.restore(); return r; } \
catch (mlexception err) { PYERROR(PyExc_OrangeKernel, err.what(), r); }


typedef struct {
  char *alias, *realName;
} TAttributeAlias;


/* We use TOrangeType instead of PyObject as a type for type definitions
   because we need some additional slots:
   - classinfo with C++ object's typeid
   - memorymark used for Orange's internal garbage collection
   - aliases for maintaining compatibility without showing old names 
*/

typedef POrange (*defaultconstrproc)(PyTypeObject *);

#ifdef _MSC_VER
  #pragma warning (disable : 4512) // assigment operator could not be generated (occurs below, due to references)
#endif

class TOrangeType { 
public:
  PyTypeObject       ot_inherited;
  const type_info   &ot_classinfo;
  defaultconstrproc  ot_defaultconstruct;
  TAttributeAlias   *ot_aliases;

  TOrangeType(const PyTypeObject &inh, const type_info &cinf, defaultconstrproc dc, TAttributeAlias *ali=NULL)
   : ot_inherited(inh), ot_classinfo(cinf), ot_defaultconstruct(dc), ot_aliases(ali)
   {}
};

#ifdef _MSC_VER
  #pragma warning (default: 4512) // assigment operator could not be generated (occurs below, due to references)
#endif

extern int noOfOrangeClasses;
extern TOrangeType *orangeClasses[];

TOrangeType *FindOrangeType(const type_info &);

// Checks whether the object (or type) is one of orange's types
bool PyOrange_CheckType(PyTypeObject *);

// Ascends the hierarchy until it comes to a class that is from orange's hierarchy
TOrangeType *PyOrange_OrangeBaseClass(PyTypeObject *);

bool SetAttr_FromDict(PyObject *self, PyObject *dict);

/* Do we need something like this?! Here?! Not in root.hpp?!
void raiseWarning(PyObject *warnType, const char *s, ...);
void raiseError(PyObject *excType, const char *s, ...);
*/

#endif
