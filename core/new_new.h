#if !defined(NEW_NEW_H)
#define NEW_NEW_H

#include <new.h>
#include <stddef.h>


#ifdef BORLAND
   #include <alloc.h>
#else
   #include <malloc.h>
#endif

   #define ALLOC  malloc
   #define FREE   free
   void  operator delete(void* ptr);
   void* operator new(size_t size);
#endif



