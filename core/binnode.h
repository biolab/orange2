#if !defined(BINNODE_H)
#define BINNODE_H

#include "expr.h"
#include "constrct.h"
#include "contain.h"

enum nodeType {continuousAttribute=0, discreteAttribute=1, leaf=2}  ;

// node for binary tree
struct binnode
{
    nodeType Identification ;
    expr Model ;
    construct Construct ;
    double weight, weightLeft ;
    marray<int> DTrain ;
    marray<double> NAcontValue ;
    marray<int> NAdiscValue ;
    marray<double> Classify ;      // weights of all classes in node
    int majorClass ;
 
    binnode *left,*right;

    binnode(void) { left = right = 0 ; } 
    void operator= (binnode &Source) ;
};

#endif
