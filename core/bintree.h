#if !defined(BINTREE_H)
#define BINTREE_H

#include "expr.h"
#include "constrct.h"
#include "contain.h"
#include "binnode.h"


// the basic binary tree
class bintree {
friend class featureTree ;
friend class rf ;

protected:
   binnode *root ;
   void dup(binnode *Source, binnode* &Target) ;

public:

   bintree() { root=0; }
   ~bintree() { destroy(root); }
   bintree(bintree &Copy) ;
   int operator== (bintree &) { return 0 ; }
   int operator< (bintree &) { return 0 ; }
   int operator> (bintree &) {return 0; }
   void operator= (bintree &Source) ;
   void copy(bintree &Source) ;
   void destroy(void) { destroy(root) ; root = 0 ; }
   void destroy(binnode *branch);
   int noLeaves(void) const { return noLeaves(root) ; }
   int noLeaves(binnode *branch) const;
   int degreesOfFreedom(void) const { return degreesOfFreedom(root) ; }
   int degreesOfFreedom(binnode *branch) const ;
   


};


#endif

