
#include "general.h"
#ifdef DEBUG_NEW
#include "new_new.h"
#include "error.h"
 
 const long MAX_SET = 100000; // high limit

void* Set[MAX_SET];
long int    SetSize = 0;

void Add(void *ptr)
{
  if ( SetSize > MAX_SET )  // overflow
  {
     error("NEW_NEW::Add","pointer table size overflow") ;
//     exit( ERR_ALLOC_FULL );
  }
  else
     Set[SetSize++] = ptr;
}

boolean Member(void *ptr)
{
  int i;
  boolean IsMember = FALSE;
  for (i=0; i<SetSize; i++) {
      if (ptr == Set[i]) {
         Set[i] = Set[--SetSize];  // zbrisemo referencirani kazalec
         IsMember = TRUE;
         break; // for loop
      }
  }
  return IsMember;
}

/* ---------------------------------------------------------------------- */
// Zaseganje pomnilnika

void *forced_new(unsigned long size)
{
  if (size<=0)
    return 0;
  if (size > 10000)
     error("NEW_NEW::forced_new","allocating very large block") ;

  void *ptr = ALLOC(size);
  if (!ptr)
  {
     error("NEW_NEW::forced_new","not enough memory available") ;
     // exit( ERR_ALLOC_NULL );
  }
  Add(ptr);
  return ptr;
}


void* operator new(size_t size)
{
  return forced_new(size);
}


void  operator delete(void* ptr)
{

  if (!ptr)
  {
//     error("NEW_NEW::oprator delete""warning, deallocating null pointer") ;
     return ;
  }
  else
    if (!Member(ptr))
    {
       error("NEW_NEW::operator delete","deallocating unexisting pointer") ;
       return;
    }

  FREE(ptr);

}


#endif

