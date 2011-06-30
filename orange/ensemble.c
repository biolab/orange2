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

/*  This file is a part of dynamic library that staticaly links with C4.5.
    To compile it, download C4.5 (R8) from http://www.cse.unsw.edu.au/~quinlan/,
    copy this file and buildc45.py to C45's directory R8/Src and run buildc45.py.

    The script should work on Windows, Linux and Mac OSX. For other systems, you
    will need to modify buildc45.py and Orange's c4.5.cpp; contact me for help or,
    if you manage to do it yourself, send me the patches if you want to make your
    port available.


    This file redefines malloc, calloc, free and cfree to record the (de)allocations,
    so that any unreleased memory can be freed after learning is done.
    
    The library exports a function that calls C4.5's learning, and a function
    for clearing the memory afterwards.

    Finally, this file contains functions (O_Iterate, O_OneTree and O_BestTree) that
    are based on functions Iterate, OneTree and BestTree from Ross Quinlan's C4.5.
    They had to be rewritten to avoid the unnecessary printouts (I don't remember
    why haven't I simply redefined printf to do nothing). Besides, I have removed
    from them storing the data that C4.5 needed only for output, not for classification.
    In other ways, the functions should be equivalent to the originals.
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
    allocationsSize = 2048;
    emptySlot = allocations = malloc(allocationsSize * sizeof(void *));
    lastAllocation = allocations + allocationsSize;
  }

  else if (emptySlot == lastAllocation) {
      nal = allocationsSize * 1.5;
      allocations = realloc(allocations, nal * sizeof(void *));
      emptySlot = allocations + allocationsSize;
      allocationsSize = nal;
      lastAllocation = allocations + allocationsSize;
  }

  *emptySlot = callo ? calloc(size, elsize + 128) : malloc(size + 128);
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


void guarded_free(void *ptr)
{ 
/* It seems that there are callocs the manage to
   avoid being allocated through my macros, so I
   cannot rely on guarded_collect but must also
   define  guarded_free.

   (Namely there's one in prune.c that allocates
    LocalClassDist) */

  void **slot = findSlot(ptr);
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
#define free(p) guarded_free((p))
#define cfree(p) guarded_free((p))

#define realloc(p,x) This is defined just to make sure that we have not missed any reallocs in C4.5 (we found none)


void Error(int l, char *s, char *t)
{}

#include <math.h>
#include "types.h"

short MaxAtt, MaxClass, MaxDiscrVal;
ItemNo MaxItem;
Description *Item;
DiscrValue *MaxAttVal;
char *SpecialStatus, **ClassName, **AttName, ***AttValName;

#define PrintConfusionMatrix(x) // make besttree.c happy

#include "besttree.c"
#include "classify.c"
#include "contin.c"
#include "discr.c"
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
Tree		*Pruned = NULL;
Boolean		AllKnown;


/* Need to redefine these to avoid printouts */


void Swap();
ClassNo Category();
Tree FormTree();

Tree O_Iterate(ItemNo Window, ItemNo IncExceptions)
{
  Tree Classifier, BestClassifier = Nil;
  ItemNo i, Errors, TotalErrors, BestTotalErrors = MaxItem+1, Exceptions, Additions;
  ClassNo Assigned;

  do {
    InitialiseWeights();
    AllKnown = true;
    Classifier = FormTree(0, Window-1);

    Errors = Round(Classifier->Errors);

    Exceptions = Window;
    ForEach(i, Window, MaxItem) {
      Assigned = Category(Item[i], Classifier);
      if (Assigned != Class(Item[i]) ) {
        Swap(Exceptions, i);
        Exceptions++;
	    }
	  }

    Exceptions -= Window;
    TotalErrors = Errors + Exceptions;

    if (!BestClassifier || (TotalErrors < BestTotalErrors)) {
      if (BestClassifier)
        ReleaseTree(BestClassifier);
      BestClassifier = Classifier;
      BestTotalErrors = TotalErrors;
    }
    else 
	    ReleaseTree(Classifier);

    Additions = Min(Exceptions, IncExceptions);
    Window = Min(Window + Max(Additions, Exceptions / 2), MaxItem + 1);
  }
  while (Exceptions);

  return BestClassifier;
}


Tree O_OneTree(char prune)
{
  Tree tree;

  InitialiseTreeData();
  InitialiseWeights();

  AllKnown = true;
  tree = FormTree(0, MaxItem);

  if (prune)
    Prune(tree);

  return tree;
}


Tree O_BestTree(int trials, int window, int increment, char prune)
{
  short t;
  
  Tree best = NULL, tree;

  InitialiseTreeData();

  TargetClassFreq = (ItemNo *) calloc(MaxClass+1, sizeof(ItemNo));

  if (!window)
    window = (int)(Max(2 * sqrt(MaxItem+1.0), (MaxItem+1) / 5));

  if (!increment)
    increment = Max(window / 5, 1);

  FormTarget(window);

  ForEach(t, 0, trials-1 ) {
    FormInitialWindow();
    tree = O_Iterate(window, increment);

    if (prune)
      Prune(tree);

    if ( !best || tree->Errors < best->Errors )
      best = tree;
    else
      ReleaseTree(tree);
  }

  return best;
}


C45_API Tree learn(int trials, char gainRatio, char subset, char batch, char probThresh, int minObjs, int window, int increment, float cf, char prune)
{
  Tree best;
 
  /* These get initialized to NULL when c45 is loaded.
     Since garbage collection frees their memory, we reinitialized
     them to force c45 to allocate the memory again. */

  ClassSum = NULL;
  PossibleValues = NULL;

  VERBOSITY  = 0;

  GAINRATIO  = gainRatio;
  SUBSET     = subset;
  MINOBJS    = minObjs;
  CF         = cf;
  
  best = batch ? O_OneTree(prune) : O_BestTree(trials, window, increment, prune);
  if (probThresh)
    SoftenThresh(best);

  return best;
}


Tree cpp_learn(int trials, char gainRatio, char subset, char batch, char probThresh, int minObjs, int window, int increment, float cf, char prune)
{ return learn(trials, gainRatio, subset, batch, probThresh, minObjs, window, increment, cf, prune); }
