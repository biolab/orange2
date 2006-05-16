// ********************************************************************
// *
// *   Name:          module constrct
// *
// *   Description:      deals with constructs 
// *
// *********************************************************************
#include <float.h>
#include <string.h>

#include "estimator.h"
#include "constrct.h"
#include "ftree.h"
#include "error.h"
#include "options.h"

extern featureTree* gFT ;
extern Options *opt ;



// ********************************************************************
// *
// *             construct
// *             --------
// *
// *         constructor of class construct 
// *
// *********************************************************************
construct::construct(construct &Copy)
{
   root = 0 ;
   copy(Copy) ;
}


// ********************************************************************
// *
// *             ~construct
// *             ---------
// *
// *         destructor of class construct 
// *
// *********************************************************************
construct::~construct()
{
   if (root)
     destroy(root) ;
}



// ********************************************************************
// *
// *             destroy
// *             -------
// *
// *     the eraser of class construct 
// *
// *********************************************************************
void construct::destroy(void)
{
   if (root)
     destroy(root) ;
   root = 0 ;
}

// ********************************************************************
// *
// *             destroy
// *             -------
// *
// *     the eraser of class construct 
// *
// *********************************************************************
void construct::destroy(constructNode *node)
{
    if (node->left)
      destroy(node->left) ;
    if (node->right)
      destroy(node->right) ;

    delete node ;

}

// ********************************************************************
// *
// *             copy   and operator =
// *             ----------------------
// *
// *     the copier of class construct 
// *
// *********************************************************************
void construct::copy(construct &Source)
{
   if (root)
     destroy(root) ;
   if (Source.root)
     dup(Source.root,root) ;
   else
     root = 0 ;
 //  TreeNode = Source.TreeNode ;
   countType = Source.countType ;
   compositionType = Source.compositionType ;
   leftValues = Source.leftValues ;
   splitValue = Source.splitValue ;
   noValues = Source.noValues ;
}

void construct::operator= (construct &X) 
{
   copy(X) ;
}


// **********************************************************************
//
//                      dup
//                      -------
//
//      duplicates source construct node into target
//
// **********************************************************************
void construct::dup(const constructNode *Source, PconstructNode &Target)
{
    Target = new constructNode ;
    Target->nodeType = Source->nodeType ;
    Target->attrIdx = Source->attrIdx ;
    Target->valueIdx = Source->valueIdx ;
    Target->lowerBoundary = Source->lowerBoundary ;
    Target->upperBoundary = Source->upperBoundary ;

    if (Source->left)
      dup(Source->left, Target->left) ;
    else
      Target->left = 0 ;
    if (Source->right)
      dup(Source->right, Target->right ) ;
    else
      Target->right = 0 ;
}




// ********************************************************************
// *
// *             descriptionString
// *             -----------------
// *
// *    fills in the description of construct 
// *
// *********************************************************************
void construct::descriptionString(char* const Str)
{
   char *dscrStr = description(root) ;

 
   switch(countType)
   {
      case aCONTINUOUS:
           //sprintf(Str,"[%d] %s <= %f", splitEstimator, dscrStr, splitValue) ;
           sprintf(Str,"%s <= %f", dscrStr, splitValue) ;
           break ;
      case aDISCRETE:
           //sprintf(Str, "[%d] %s", splitEstimator, dscrStr) ;
           sprintf(Str, "%s", dscrStr) ;
           if (compositionType == cSINGLEattribute)
           { 
             strcat(Str, "= (") ; 
             int pos = 1 ;
             // first value
             while (pos < leftValues.len() && (leftValues[pos] == FALSE))
                pos++ ;
             if (pos < leftValues.len())
                strcat(Str, gFT->AttrDesc[gFT->DiscIdx[root->attrIdx]].ValueName[pos-1]) ;
             else
                error("construct::descriptionString","invalid binarization detected") ;
             for (int i=pos+1 ; i < leftValues.len() ; i++)
                if (leftValues[i])
                {
                   strcat(Str, " | ") ;
                   strcat(Str, gFT->AttrDesc[gFT->DiscIdx[root->attrIdx]].ValueName[i-1]) ;
                }
             strcat(Str,")") ;
           }
           break ;
               
      default: error("construct::descriptionString","invalid count type") ;
   } 
   delete [] dscrStr ;
}


// ********************************************************************
// *
// *             description
// *             ------------
// *
// *    recursively collects the description of construct 
// *
// *********************************************************************
char* construct::description(constructNode *Node) 
{
   char *Str = new char[MaxFeatureStrLen] ;
   switch(Node->nodeType)
   {
      case cnDISCattrValue:
           sprintf(Str, "(%s = %s)", gFT->AttrDesc[gFT->DiscIdx[Node->attrIdx]].AttributeName,
                                   gFT->AttrDesc[gFT->DiscIdx[Node->attrIdx]].ValueName[Node->valueIdx-1] ) ;
           break ;          
      case cnCONTattrValue:
           if (Node->lowerBoundary == -FLT_MAX)
              sprintf(Str, "(%s <= %.3f)", gFT->AttrDesc[gFT->ContIdx[Node->attrIdx]].AttributeName,
                                           Node->upperBoundary ) ;
           else
             if (Node->upperBoundary == FLT_MAX)
                sprintf(Str, "(%s > %.3f)", gFT->AttrDesc[gFT->ContIdx[Node->attrIdx]].AttributeName,
                                           Node->lowerBoundary ) ;
             else
                sprintf(Str, "(%.3f < %s <= %.3f)", Node->lowerBoundary,
                                                    gFT->AttrDesc[gFT->ContIdx[Node->attrIdx]].AttributeName,
                                                    Node->upperBoundary ) ;
           break ;
      
      case cnCONTattribute:
           sprintf(Str,"%s", gFT->AttrDesc[gFT->ContIdx[Node->attrIdx]].AttributeName) ;
           break ;
      
      case cnDISCattribute:
           sprintf(Str, "%s", gFT->AttrDesc[gFT->DiscIdx[Node->attrIdx]].AttributeName) ;
           break ;

      case cnAND:
         {
            char *leftStr = description(Node->left) ;
            char *rightStr = description(Node->right) ;
            sprintf(Str, "%s & %s", leftStr, rightStr) ;
            delete [] leftStr ;
            delete [] rightStr ;
         }
         break ;
      case cnTIMES:
         {
            char *leftStr = description(Node->left) ;
            char *rightStr = description(Node->right) ;
            sprintf(Str, "%s * %s", leftStr, rightStr) ;
            delete [] leftStr ;
            delete [] rightStr ;
         }
         break ;
      case cnPLUS:
         {
            char *leftStr = description(Node->left) ;
            char *rightStr = description(Node->right) ;
            sprintf(Str, "%s + %s", leftStr, rightStr) ;
            delete [] leftStr ;
            delete [] rightStr ;
         }
         break ;

      default: 
            error("construct::description","invalid type of node") ;
            strcpy(Str, "ERROR(construct::description)") ;
   }
   return Str ;
}



// ********************************************************************
// *
// *             degreesOfFreedom
// *             ----------------
// *
// *     return the number of building attrubutes in construct 
// *
// *********************************************************************
int construct::degreesOfFreedom(void) 
{
   switch (compositionType)
   {
      case cSINGLEattribute:
           return 1 ;
      case cCONJUNCTION:
      case cSUM:
      case cPRODUCT:
           return degreesOfFreedom(root) ;
           
      default: 
           error("construct::degreesOfFreedom","invalid composition") ;
           return 0 ;
   } 

}


// ********************************************************************
// *
// *             degreesOfFreedom
// *             ----------------
// *
// *     return the number of building attrubutes in construct 
// *
// *********************************************************************
int construct::degreesOfFreedom(constructNode *Node) 
{
   switch (Node->nodeType)
   {
      case cnDISCattrValue:
      case cnCONTattrValue:
      case cnDISCattribute:
      case cnCONTattribute:
                         return 1 ;

      case cnAND:
      case cnTIMES:
      case cnPLUS:
                return degreesOfFreedom(Node->left) + 
                       degreesOfFreedom(Node->right) ;
      default: 
           error("construct::degreesOfFreedom","invalid node type") ;
           return 0 ;
   } 

}



// ********************************************************************
// *
// *             continuousValue/1
// *             ---------------
// *
// *     returns the value of continuous construct
// *
// *********************************************************************/
double construct::continuousValue(mmatrix<int> &DiscData, mmatrix<double> &ContData, int caseIdx) 
{
#ifdef DEBUG
  if (countType != aCONTINUOUS) 
    error("construct::continuousValue", "invalid count of construct") ;
#endif

  switch (compositionType)
  {
     case cSINGLEattribute:
       return ContData(caseIdx, root->attrIdx) ;

     case cSUM:
     case cPRODUCT:
        return continuousValue(DiscData, ContData, caseIdx, root) ;
     
     default:
        error("construct::continuousValue", "invalid composition type detected") ;    
        return -FLT_MAX ;
  }

}



// ********************************************************************
// *
// *             continuousValue/2
// *             ---------------
// *
// *     returns the value of continuous construct 
// *
// *********************************************************************/
double construct::continuousValue(mmatrix<int> &DiscData, mmatrix<double> &ContData, int caseIdx, constructNode* Node) 
{
   switch (Node->nodeType)
   {
     case cnCONTattribute:
        return ContData(caseIdx, Node->attrIdx) ;
     case cnTIMES:
       {
         double leftValue, rightValue ;
         leftValue = continuousValue(DiscData, ContData, caseIdx, Node->left) ;
         rightValue = continuousValue(DiscData, ContData, caseIdx, Node->right) ;
         if (leftValue == NAcont || rightValue == NAcont)
           return NAcont ;
         return leftValue*rightValue ; 
       } 
     case cnPLUS:
       {
         double leftValue, rightValue ;
         leftValue = continuousValue(DiscData, ContData, caseIdx, Node->left) ;
         rightValue = continuousValue(DiscData, ContData, caseIdx, Node->right) ;
         if (leftValue == NAcont || rightValue == NAcont)
           return NAcont ;
         return leftValue+rightValue ; 
       } 
     default: 
        error("construct::continuousValue/2", "invalid node type") ;
        return NAcont ;
  }
}


// ********************************************************************
// *
// *             discreteValue/1
// *             -------------
// *
// *     returns the value of discrete construct
// *
// *********************************************************************/
int construct::discreteValue(mmatrix<int> &DiscData, mmatrix<double> &ContData, int caseIdx)
{
#ifdef DEBUG
  if (countType != aDISCRETE) 
    error("construct::discreteValue", "invalid count of construct") ;
#endif

  
  switch (compositionType)
  {
     case cSINGLEattribute:
       return DiscData(caseIdx, root->attrIdx) ;
     case cCONJUNCTION:
       return discreteValue(DiscData, ContData, caseIdx, root) ;

     default:
       error("construct::discreteValue/1", "invalid composition type of construct") ;
       return NAdisc ;
  }
}


// ********************************************************************
// *
// *             discreteValue/2
// *             -------------
// *
// *     returns the value of discrete construct 
// *
// *********************************************************************/
int construct::discreteValue(mmatrix<int> &DiscData, mmatrix<double> &ContData, int caseIdx, constructNode* Node) 
{
   int discValue, anotherDiscValue ;
   double contValue ;
   switch (Node->nodeType)
   {
     case cnDISCattribute:
        return DiscData(caseIdx, Node->attrIdx) ;
     case cnDISCattrValue:
        discValue = DiscData(caseIdx, Node->attrIdx) ;
        if (discValue == NAdisc)
           return NAdisc ;
        if (discValue == Node->valueIdx)
          return 1 ;
        else 
          return 2 ;
     case cnCONTattrValue:
        contValue = ContData(caseIdx, Node->attrIdx) ;
        if (contValue == NAcont)
           return NAdisc ;
        if (contValue > Node->lowerBoundary &&
            contValue <= Node->upperBoundary)
          return 1 ;
        else 
          return 2 ;
     case cnAND:
        discValue = discreteValue(DiscData, ContData, caseIdx, Node->left) ;
        anotherDiscValue = discreteValue(DiscData, ContData, caseIdx, Node->right) ;
        if (discValue == NAdisc || anotherDiscValue == NAdisc)
           return NAdisc ;
        if (discValue == 1 && anotherDiscValue == 1)
           return 1 ;
        else
           return 2 ;

     default: 
        error("construct::discreteValue/2", "invalid node type") ;
        return NAdisc ;
  }
}


// ********************************************************************
// *
// *             createSingle
// *             ------------
// *
// *     creates traditional one-attribute construct 
// *
// *********************************************************************
void construct::createSingle(int bestIdx, attributeCount count) 
{
   // warning: for discrete attributes number of values should be set by caller !!

   destroy() ;
   countType = count ;
   compositionType = cSINGLEattribute ;
   root = new constructNode ;
   root->left = root->right=0 ;
   root->attrIdx = bestIdx ;
   
   switch (countType)
   {
      case aCONTINUOUS:
        root->nodeType = cnCONTattribute ;
        break ;
      
      case aDISCRETE:
        root->nodeType = cnDISCattribute ;
        break ;

      default: error("construct::singleAttribute","invalid count type") ;
   }

}


// ********************************************************************
// *
// *             conjoin
// *             ------------
// *
// *     makes conjunction of two constructs
// *
// *********************************************************************
void construct::Conjoin(construct &First, construct &Second)
{
   destroy() ;
   countType = aDISCRETE ;
   compositionType = cCONJUNCTION ;
   root = new constructNode ;
   root->nodeType = cnAND ;
   dup(First.root, root->left) ;
   dup(Second.root, root->right) ;
}


// ********************************************************************
// *
// *             add
// *             ----
// *
// *     makes sum of two constructs
// *
// *********************************************************************
void construct::add(construct &First, construct &Second)
{
   destroy() ;
   countType =  aCONTINUOUS;
   compositionType = cSUM ;
   root = new constructNode ;
   root->nodeType = cnPLUS ;
   dup(First.root, root->left) ;
   dup(Second.root, root->right) ;
}



// ********************************************************************
// *
// *             multiply
// *             --------
// *
// *     makes product of two constructs
// *
// *********************************************************************
void construct::multiply(construct &First, construct &Second)
{
   destroy() ;
   countType =  aCONTINUOUS;
   compositionType = cPRODUCT ;
   root = new constructNode ;
   root->nodeType = cnTIMES ;
   dup(First.root, root->left) ;
   dup(Second.root, root->right) ;
}


// ********************************************************************
// *
// *            containsAttribute 
// *            -----------------
// *
// *     checks if construct contains attribute of
// *         given attribute's value
// *
// *********************************************************************
boolean construct::containsAttribute(construct &AttrConstruct)
{
#if defined(DEBUG)
   if (AttrConstruct.root->left != 0 || AttrConstruct.root->right != 0 ||
       (AttrConstruct.root->nodeType != cnCONTattrValue && 
        AttrConstruct.root->nodeType != cnDISCattrValue &&
        AttrConstruct.root->nodeType != cnCONTattribute &&
        AttrConstruct.root->nodeType != cnDISCattribute) )
      error("construct::containsAttribute", "unexpected construct was given as input") ;
#endif

    if (root)
       return containsAttribute(root, AttrConstruct.root->attrIdx) ;
    else
       return FALSE ;
}


// ********************************************************************
// *
// *            containsAttribute 
// *            -----------------
// *
// *     checks if construct node or its subnodes contain 
// *            given attribute
// *
// *********************************************************************
boolean construct::containsAttribute(constructNode *Node, int attributeIdx)
{
   if (Node->attrIdx == attributeIdx)
      return TRUE ;
   
   if (Node->left)
      if (containsAttribute(Node->left, attributeIdx))
         return TRUE ;
   
   if (Node->right)
     return containsAttribute(Node->right, attributeIdx) ;
        
   return FALSE ;
   
}



// ********************************************************************
// *
// *            operator== 
// *            -----------
// *
// *     checks if two constructs are the same
// *
// *********************************************************************
int construct::operator== (construct &X) 
{
   if (countType != X.countType || compositionType != X.compositionType)
      return 0 ;
   switch (compositionType)
   {
     case cSINGLEattribute:
       if (root->nodeType != X.root->nodeType || 
           root->attrIdx != X.root->attrIdx)
          return 0 ;
       else
          return 1;
     case  cCONJUNCTION:
     {
       
       int noConjs = degreesOfFreedom() ;
       int XnoConjs = X.degreesOfFreedom() ;
       if (noConjs != XnoConjs) 
         return 0 ;
       marray<int> discAttrIdxs(noConjs), AttrVals(noConjs), contAttrIdxs(noConjs) ;
       marray<int> XdiscAttrIdxs(XnoConjs), XAttrVals(XnoConjs), XcontAttrIdxs(XnoConjs) ;
       marray<double> lowerBndys(noConjs), upperBndys(noConjs) ;
       marray<double> XlowerBndys(XnoConjs), XupperBndys(XnoConjs) ;

       flattenConjunct(discAttrIdxs, AttrVals, contAttrIdxs, lowerBndys, upperBndys) ;
       X.flattenConjunct(XdiscAttrIdxs, XAttrVals, XcontAttrIdxs, XlowerBndys, XupperBndys) ;
       
       if (discAttrIdxs.filled() != XdiscAttrIdxs.filled() || 
           contAttrIdxs.filled() != XcontAttrIdxs.filled() )
         return 0 ;
       
       int i, j ;
       boolean noSuchComponent ;
       // check attribute values of discrete attributes
       for (i=0 ; i < discAttrIdxs.filled() ; i++)
       {
         noSuchComponent = TRUE ;
         for (j=0 ; j < XdiscAttrIdxs.filled() ; j++)
            if (discAttrIdxs[i] == XdiscAttrIdxs[j] && AttrVals[i] == XAttrVals[j])
            {
               noSuchComponent = FALSE ;
               break ;
            }
         if (noSuchComponent)
            return 0 ;
       }

       // check intervals of continuous attributes
       for (i=0 ; i < contAttrIdxs.filled() ; i++)
       {
         noSuchComponent = TRUE ;
         for (j=0 ; j < XcontAttrIdxs.filled() ; j++)
            if (contAttrIdxs[i] == XcontAttrIdxs[j] && lowerBndys[i] == XlowerBndys[j]
                && upperBndys[i] == XupperBndys[j])
            {
               noSuchComponent = FALSE ;
               break ;
            }
         if (noSuchComponent)
            return 0 ;
       }
       // they are the same
       return 1; 
     }
     case cSUM:
     case cPRODUCT:
     {
       
       int noConts = degreesOfFreedom() ;
       int XnoConts = X.degreesOfFreedom() ;
       if (noConts != XnoConts) 
         return 0 ;
       marray<int> contAttrIdxs(noConts), XcontAttrIdxs(XnoConts) ;

       flattenContConstruct(contAttrIdxs) ;
       X.flattenContConstruct(XcontAttrIdxs) ;
       
       if (contAttrIdxs.filled() != XcontAttrIdxs.filled() )
         return 0 ;
       
       int i, j ;
       boolean noSuchComponent ;
       // check continuous attributes
       for (i=0 ; i < contAttrIdxs.filled() ; i++)
       {
         noSuchComponent = TRUE ;
         for (j=0 ; j < XcontAttrIdxs.filled() ; j++)
           if (contAttrIdxs[i] == XcontAttrIdxs[j]) 
           {
              noSuchComponent = FALSE ;
              break ;
           }
         if (noSuchComponent)
            return 0 ;
       }
       // they are the same
       return 1; 
     }

     case cXofN:
     default:
       error("construct::operator==", "invalid composition type") ;
       return 0 ;
   }
}


// ********************************************************************
// *
// *            flattenConjunct 
// *            --------------
// *
// *     gives flat representation of conjunctive construct 
// *
// *********************************************************************
void construct::flattenConjunct(marray<int> &discAttrIdxs, marray<int> &AttrVals, 
                     marray<int> &contAttrIdxs, marray<double> &lowerBndys, marray<double> &upperBndys) 
{
  // asumption: there is enough space in array to hold values
  discAttrIdxs.setFilled(0) ;
  AttrVals.setFilled(0) ;
  contAttrIdxs.setFilled(0) ;
  lowerBndys.setFilled(0) ;
  upperBndys.setFilled(0) ;

  if (root) 
     flattenConjunct(root, discAttrIdxs, AttrVals, contAttrIdxs, lowerBndys, upperBndys) ;
}


// ********************************************************************
// *
// *            flattenConjunct 
// *            --------------
// *
// *     gives flat representation of conjunctive construct 
// *
// *********************************************************************
void construct::flattenConjunct(constructNode *Node, marray<int> &discAttrIdxs, marray<int> &AttrVals, 
                     marray<int> &contAttrIdxs, marray<double> &lowerBndys, marray<double> &upperBndys) 
{
   switch (Node->nodeType)
   {
     case cnAND:
        if (Node->left)
           flattenConjunct(Node->left, discAttrIdxs, AttrVals, contAttrIdxs, lowerBndys, upperBndys) ;
        if (Node->right)
           flattenConjunct(Node->right, discAttrIdxs, AttrVals, contAttrIdxs, lowerBndys, upperBndys) ;
        break ; 
     case cnCONTattrValue:
        contAttrIdxs.addEnd(Node->attrIdx) ;
        lowerBndys.addEnd(Node->lowerBoundary) ;
        upperBndys.addEnd(Node->upperBoundary) ;
        break ;
     case cnDISCattrValue:
        discAttrIdxs.addEnd(Node->attrIdx) ;
        AttrVals.addEnd(Node->valueIdx) ;
        break ;
     default:
        error("construct::flattenConjunct", "unexpected node type detected") ;
   }
}



// ********************************************************************
// *
// *            flattenContConstruct 
// *            --------------------
// *
// *     gives flat representation of summary or product
// *
// *********************************************************************
void construct::flattenContConstruct(marray<int> &contAttrIdxs) 
{
  // asumption: there is enough space in array to hold values
  contAttrIdxs.setFilled(0) ;

  if (root) 
     flattenContConstruct(root, contAttrIdxs) ;
}


// ********************************************************************
// *
// *            flattenContConstruct
// *            --------------------
// *
// *     gives flat representation of summary or product
// *
// *********************************************************************
void construct::flattenContConstruct(constructNode *Node, marray<int> &contAttrIdxs) 
{
   switch (Node->nodeType)
   {
     case cnTIMES:
     case cnPLUS: 
       if (Node->left)
           flattenContConstruct(Node->left, contAttrIdxs) ;
        if (Node->right)
           flattenContConstruct(Node->right, contAttrIdxs) ;
        break ; 
     case cnCONTattribute:
        contAttrIdxs.addEnd(Node->attrIdx) ;
        break ;
     default:
        error("construct::flattenContConstruct", "unexpected node type detected") ;
   }
}


// ********************************************************************
// *
// *            mdlAux/1 
// *            --------
// *
// *     computes part of the mdl code len for  construct 
// *
// *********************************************************************
double construct::mdlAux() 
{
   switch (compositionType)
   {
      case cCONJUNCTION:
      case cSUM:
      case cPRODUCT:
         return mdlAux(root) ;
        
      case cSINGLEattribute:
      default:  
         error("construct::mdlAux", "unexpected behaviour") ;
         return 0.0 ;
   }
}


// ********************************************************************
// *
// *            mdlAux/2
// *            --------
// *
// *     computes part of the mdl code len for  construct 
// *
// *********************************************************************
double construct::mdlAux(constructNode *Node) 
{
   switch(Node->nodeType)
   { 
      case cnAND:
      case cnPLUS:
      case cnTIMES:
         return mdlAux(Node->left) + mdlAux(Node->right) ;

      case cnCONTattribute: // summing and multiplying
         return log2(gFT->NoContinuous-1) ;
      
      case cnCONTattrValue:
      {
        double intValue = gFT->valueInterval[Node->attrIdx]/opt->mdlErrorPrecision ;
        if (intValue < 1.0)
          intValue = 1.0 ;
        return  log2(gFT->NoAttr) + 2*log2(intValue) ;
      }
      case  cnDISCattrValue:
        return log2(gFT->NoAttr) +
               log2(gFT->AttrDesc[gFT->DiscIdx[Node->attrIdx]].NoValues) ;

      case cnDISCattribute:
      default: 
         error("construct::mdlAux", "unexpected use") ;
         return 0.0 ;
   }
}


// ********************************************************************
// *
// *            mdlConstructCode
// *            ----------------
// *
// *     computes MDL code len for  construct 
// *
// *********************************************************************
double construct::mdlConstructCode() 
{
   double code = log2(no1bits(opt->constructionMode)) ;
   switch (compositionType)
   {
     case cSINGLEattribute:
          code += log2(gFT->NoAttr) ;
          if (countType == aDISCRETE)
          {
//#if defined(DEBUG)
//            if ( ! leftValues.defined() )
//               error("construct::mdlConstructCode", "unexpected form of a call") ;
//#endif
            marray<double> Multinom(2, 0.0) ;  
            for (int i=1 ; i < leftValues.len() ; i++)
              if (leftValues[i]) 
                Multinom[0] += 1.0 ;
            Multinom[1] = leftValues.len() - 1.0 - Multinom[0];
            Multinom.setFilled(2)  ;
            code += multinomLog2(Multinom) ;
          }
          else
          {
            double intValue = gFT->valueInterval[root->attrIdx]/opt->mdlModelPrecision ;
            if (intValue < 1.0)
              intValue = 1.0 ;
            code += log2(intValue) ;
          }
          break ;       
     case cCONJUNCTION:
     case cSUM:
     case cPRODUCT:
     case cXofN:
         code += log2(degreesOfFreedom()) ;
         code += mdlAux() ;       
         break ;
     default:
         error("construct::mdlConstructCode","construct has unexpected composition") ;
   }
   return code ;
}

