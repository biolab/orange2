#ifndef __GLOBALS_HPP
#define __GLOBALS_HPP

#include "pyxtract_macros.hpp"

#include "c2py.hpp"
#include "root.hpp"

extern PyObject *orangeModule;

extern PyObject *PyExc_OrangeKernel, *PyExc_OrangeAttributeWarning, *PyExc_OrangeWarning, *PyExc_OrangeCompatibilityWarning, *PyExc_OrangeKernelWarning;

#define PyTRY try {

#define PYNULL ((PyObject *)NULL)
#define PyCATCH   PyCATCH_r(PYNULL)
#define PyCATCH_1 PyCATCH_r(-1)

#define PyCATCH_r(r) \
  } \
catch (pyexception err)   { err.restore(); return r; } \
catch (mlexception err) { PYERROR(PyExc_OrangeKernel, err.what(), r); }

void raiseWarning(bool, const char *s);
bool raiseWarning(PyObject *warnType, const char *s, ...);
bool raiseCompatibilityWarning(const char *s, ...);
/*
void raiseError(PyObject *excType, const char *s, ...);
*/


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
  char             **ot_constructorkeywords;
  TAttributeAlias   *ot_aliases;

  TOrangeType(const PyTypeObject &inh, const type_info &cinf, defaultconstrproc dc, char **ck = NULL, TAttributeAlias *ali = NULL)
   : ot_inherited(inh), ot_classinfo(cinf), ot_defaultconstruct(dc), ot_constructorkeywords(ck), ot_aliases(ali)
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

bool SetAttr_FromDict(PyObject *self, PyObject *dict, bool fromInit = false);



#endif