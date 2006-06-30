#ifndef __JIT_LINKER_HPP
#define __JIT_LINKER_HPP

#include <stdlib.h>

typedef void (*TDefaultFunc)(...);

typedef struct {
  void **address;
  char *funcname;
} TJitLink;

int jit_link(const char *dllname, TJitLink *functions, TDefaultFunc deffunc = NULL);

#endif
