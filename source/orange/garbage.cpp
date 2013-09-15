#include "garbage.hpp"
#include <string>
#include <vector>
#include <map>
using namespace std;


#ifdef _MSC_VER
 #pragma warning (disable : 4290)

#else

#include <cxxabi.h>

char *demangled = NULL;

char *demangle(const type_info &type)
{ if (demangled) {
    mldelete demangled;
    demangled = NULL;
  }
   
  int status;
  char *abidemangle = abi::__cxa_demangle(type.name(), 0, 0, &status);
  if (!status) {
    demangled = mlnew char[strlen(abidemangle)+1];
    strcpy(demangled, abidemangle);
  }
  return demangled;
}
#endif


/*void floatfloat_mapdestructor(void *x) { mldelete (map<float, float> *) x; }
template<>
TGCCounterNML<map<float, float> >::TDestructor GCPtrNML<map<float, float> >::destructor = floatfloat_mapdestructor;

typedef TGCCounterNML<int> TPyNotOrange;

void NotOrange_dealloc(TPyNotOrange *self)
{ if (!self->is_reference)
    self->destructor((void *)self->ptr);
}

PyTypeObject PyNotOrOrange_Type =  {
  PyObject_HEAD_INIT((_typeobject *)&PyType_Type)
  0,
  "Not Orange",
  sizeof(TPyNotOrange), 0,
  (destructor)NotOrange_dealloc,                     // tp_dealloc 
  0, 0, 0, 0,
  0,                             // tp_repr 
  0,                                 // tp_as_number 
  0, 0, 0, 0,
  0,                              // tp_str 
  0,                      // tp_getattro 
  0,                      // tp_setattro 
  0,
  Py_TPFLAGS_HAVE_CLASS | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_SEQUENCE_IN, // tp_flags 
  0, 0, 0, 0, 0, 0, 0,
  0,                                    // tp_methods 
  0, 0, 0, 0, 0, 0,
  offsetof(TPyNotOrange, notorange_dict),                  // tp_dictoffset 
};


*/
