/*********************************************************************
*   Name:              modul MDL
*
*   Description:  MDL based pruning
*
*********************************************************************/

#include <stdlib.h>

#include "utils.h"
#include "ftree.h"
#include "options.h"

extern Options *opt ;

//************************************************************
//
//                 mPrune
//                 -------
//
//     prune regression tree with modification of Nibbet-Bratko method using
//                       m-estimate
//
//************************************************************
double featureTree::mPrune(binnode* Node)
{

   double Es = 1.0 - (Node->Classify[Node->majorClass] + 
                      opt->mEstPruning * AttrDesc[0].valueProbability[Node->majorClass]) 
                   / (Node->weight + opt->mEstPruning) ;

   if  (Node->left == 0)  // && (Node->right == 0) ) // leaf
       // return static error
     return Es ;

   double El = mPrune(Node->left) ;
   double Er = mPrune(Node->right) ;

   double pLeft = Node->weightLeft/Node->weight ;
   double Ed = pLeft * El + (double(1.0) - pLeft) * Er ;

   if (Es <= Ed)
   {
       // prune subtrees
       destroy(Node->left) ;
       destroy(Node->right) ;

       createLeaf(Node) ;

       return Es  ;
   }
   else return Ed ;
}



double featureTree::mdlBottomUpPrune(binnode* Node)
{
   return -1.0 ;  // !!!!! not implemented yet
}


