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

/*  This file contains two functions (O_OneTree and O_BestTree) that are
    based on functions OneTree and BestTree from Ross Quinlan's C4.5.

    You need to download C4.5 (R8) from http://www.cse.unsw.edu.au/~quinlan/
    and run i2h.py before compiling this file.
*/

#include <stdio.h>

#ifndef HAS_RANDOM
long random()
{ return rand(); }
#endif

#ifdef _MSC_VER
  #define C45_API __declspec(dllexport)
  #define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

  #include <windows.h>
  #undef IGNORE

  BOOL APIENTRY DllMain( HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
  { return TRUE; }

  #ifdef _DEBUG
  #include <crtdbg.h>
  #else
  #define _ASSERT(p)
  #endif
#else
  #define C45_API

#endif

void **allocations = NULL, **emptySlot = NULL, **lastAllocation = NULL;
int allocationsSize = 0;

void *guarded_alloc(size_t size, size_t elsize, char callo)
{ 
  void **s, **t;
  int nal;

  if (!allocations) {
    allocationsSize = 1024;
    emptySlot = allocations = malloc(allocationsSize * sizeof(void *));
    lastAllocation = allocations + 1024;
  }

  else if (emptySlot == lastAllocation) {
      nal = allocationsSize * 1.5;
      allocations = realloc(allocations, nal * sizeof(void *));
      emptySlot = allocations + allocationsSize;
      allocationsSize = nal;
      lastAllocation = allocations + allocationsSize;
  }

  *emptySlot = callo ? calloc(size, elsize) : malloc(size);
  return *(emptySlot++);
}

void **findSlot(void *ptr)
{ 
  void **s;
  for(s = allocations; s != emptySlot; s++)
    if (*s == ptr)
      return s;
  return NULL;
}

void *guarded_realloc(void *ptr, size_t size)
{ 
  void **s;

  if (!ptr)
    return guarded_alloc(size, 0, 0);
    
  s = findSlot(ptr);
  // we assert that s is not NULL
  *s = realloc(ptr, size);
  return *s;
}


void guarded_free(void *ptr)
{ 
  void **slot = findSlot(ptr);
  /* I don't like this, but it seems that there are callocs
     (namely the one in prune.c that allocates LocalClassDist)
     managed to avoid being allocated through malloc. */
  _ASSERT(slot);
  *slot = NULL;
  free(ptr);
}


C45_API void guarded_collect()
{ 
  void **s;
  for(s = allocations; s != emptySlot; s++)
    if (*s)
      free(*s);

  free(allocations);
  allocations = emptySlot = lastAllocation = NULL;
  allocationsSize = 0;
}


#define calloc(x,y) guarded_alloc((x), (y), 1)
#define malloc(x) guarded_alloc((x), 0, 0)
#define realloc(p,x) guarded_realloc((p),(x))
#define free(p) guarded_free((p))
#define cfree(p) guarded_free((p))

void Error(int l, char *s, char *t)
{}

#include <math.h>
#include "types.h"

short MaxAtt, MaxClass, MaxDiscrVal = 2;
ItemNo MaxItem;
Description *Item;
DiscrValue *MaxAttVal;
char *SpecialStatus, **ClassName, **AttName, ***AttValName;

#include "besttree.c"
#include "classify.c"
#include "confmat.c"
#include "contin.c"
#include "discr.c"
#include "genlogs.c"
#include "getopt.c"
#include "info.c"
#include "prune.c"
#include "sort.c"
#include "st-thresh.c"
#include "stats.c"
#include "subset.c"
#include "trees.c"
#include "build.c"

short VERBOSITY = 0, TRIALS = 10;
Boolean GAINRATIO = true, SUBSET = false;
ItemNo MINOBJS = 2, WINDOW = 0, INCREMENT = 0;
float		CF = 0.25;

C45_API void *c45Data[] = {
  &MaxAtt, &MaxClass, &MaxDiscrVal, &MaxItem,
  &Item, &MaxAttVal, &SpecialStatus, &ClassName, &AttName,
  &AttValName
};

/* These need to be defined since we got rid of files that would otherwise define them */
char *FileName = "DF";
Tree		*Pruned;
Boolean		AllKnown;


/* Need to redefine these to avoid printouts */

void O_OneTree()
{
  InitialiseTreeData();
  InitialiseWeights();

  Raw = (Tree *) calloc(1, sizeof(Tree));
  Pruned = (Tree *) calloc(1, sizeof(Tree));

  AllKnown = true;
  Raw[0] = FormTree(0, MaxItem);

  Pruned[0] = CopyTree(Raw[0]);
  Prune(Pruned[0]);
}


short O_BestTree(int trials, int window, int increment)
{
  short t, Best=0;

  InitialiseTreeData();

  TargetClassFreq = (ItemNo *) calloc(MaxClass+1, sizeof(ItemNo));

  Raw    = (Tree *) calloc(trials, sizeof(Tree));
  Pruned = (Tree *) calloc(trials, sizeof(Tree));

  if (!window)
    window = (int)(Max(2 * sqrt(MaxItem+1.0), (MaxItem+1) / 5));

  if (!increment)
    increment = Max(window / 5, 1);

  FormTarget(window);

  ForEach(t, 0, trials-1 ) {
    FormInitialWindow();
    Raw[t] = Iterate(window, increment);

    Pruned[t] = CopyTree(Raw[t]);
    Prune(Pruned[t]);

    if ( Pruned[t]->Errors < Pruned[Best]->Errors )
      Best = t;
  }

  return Best;
}


C45_API Tree learn(int trials, char gainRatio, char subset, char batch, char probThresh, int minObjs, int window, int increment, float cf, char prune)
{
  short Best;
  
  VERBOSITY  = 0;

  GAINRATIO  = gainRatio;
  SUBSET     = subset;
  MINOBJS    = minObjs;
  CF         = cf;
  
  if (batch) {
    O_OneTree();
    Best = 0;
  }
  else
    Best = O_BestTree(trials, window, increment);

  if (probThresh)
    SoftenThresh((prune ? Pruned : Raw)[Best]);

  return (prune ? Pruned : Raw)[Best];
}


Tree cpp_learn(int trials, char gainRatio, char subset, char batch, char probThresh, int minObjs, int window, int increment, float cf, char prune)
{ return learn(trials, gainRatio, subset, batch, probThresh, minObjs, window, increment, cf, prune); }
