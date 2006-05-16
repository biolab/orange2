#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "error.h"
#include "ftree.h"
#include "mathutil.h"
#include "options.h"

extern Options *opt ;


//************************************************************
//
//                 buildModel
//                 ---------
//
//    builds model to explain the data in a node
//
//************************************************************
void featureTree::buildModel(estimation &Estimator, binnode* Node)
{
   // what kind of a model do we use in a leaf
   switch (opt->modelType)   {
       case 1:
              //  majority class value
              Node->Model.createMajority(Node->majorClass) ;
              break;
       case 2:  // k-NN
              Node->Model.createKNN() ; 
              break ;
       case 3:  // k-NN
              Node->Model.createKNNkernel() ; 
              break ;
       case 4:  // simple Bayes
              Node->Model.createSimpleBayes(Estimator, Node) ; 
              break ;
      default: error("featureTree::buildModel","invalid modelType detected") ;
   }
}

