#if !defined(RNDFOREST_H)
#define RNDFOREST_H

#include "contain.h"
#include "ftree.h"


struct IntSortRec
{
   int value ;
   int key;
   inline void operator= (IntSortRec& X) { value=X.value; key=X.key; }
   inline int operator== (IntSortRec &X) { if (key==X.key) return 1 ; else return 0 ; }
   inline int operator> (IntSortRec& X) { if (key > X.key)  return 1 ; else return 0 ; }
   inline int operator< (IntSortRec& X) { if ( key < X.key)  return 1 ; else return 0 ; }
};

struct BinNodeRec
{
   binnode *value ;
   double key;
   inline void operator= (BinNodeRec& X) { value=X.value; key=X.key; }
   inline int operator== (BinNodeRec &X) { if (key==X.key) return 1 ; else return 0 ; }
   inline int operator> (BinNodeRec& X) { if (key > X.key)  return 1 ; else return 0 ; }
   inline int operator< (BinNodeRec& X) { if ( key < X.key)  return 1 ; else return 0 ; }
};

class forestTree {
public:
   marray<int> ib ; // in bag instances
   marray<boolean> oob ; // out of bag instances
   bintree t ;
   forestTree(void) {}
   ~forestTree(void) {} 
} ;


#endif
