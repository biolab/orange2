/********************************************************************
*
*   Name:   class bintree
*
*   Description: base class, representing binary tree
*
*********************************************************************/

#include "bintree.h"


bintree::bintree(bintree &Copy)
{
   root = 0 ;
   copy(Copy) ;
}

//************************************************************
//
//                       destroy
//                       -------
//
//          recursivly destroys entire binary tree
//
//************************************************************
void bintree::destroy(binnode *branch)
{
   if (branch)
   {
      destroy(branch->left);
      destroy(branch->right);

      delete branch ;
   }
}

//************************************************************
//
//                       noLeaves
//                       --------
//
//          counts the leaves in the binary tree
//
//************************************************************
int bintree::noLeaves(binnode *branch) const
{
   if (branch->left==0) // && (branch->right==0))
      return 1 ;
   return noLeaves(branch->left) + noLeaves(branch->right) ;
}


// **********************************************************************
//
//                      copy
//                      --------
//
//      copies source tree
//
// **********************************************************************
void bintree::copy(bintree &Source)
{
   if (root)
     destroy(root) ;
   if (Source.root)
     dup(Source.root,root) ;
   else
     root = 0 ;
}


// **********************************************************************
//
//                      operator =
//                      --------
//
//      copies source tree
//
// **********************************************************************
void bintree::operator=(bintree &Source)
{
   copy (Source) ;
}


//**********************************************************************
//
//                      dup
//                      -------
//
//      duplicates source boolean expression into target
//
//**********************************************************************
void bintree::dup(binnode *Source, binnode* &Target)
{
    Target = new binnode ;
    Target = Source ;

    if (Source->left)
      dup(Source->left, Target->left) ;
    else
      Target->left = 0 ;
    if (Source->right)
      dup(Source->right, Target->right ) ;
    else
      Target->right = 0 ;
}




//************************************************************
//
//                       degreesOfFreedom
//                       ----------------
//
//          counts the occurences of all attributes
//
//************************************************************
int bintree::degreesOfFreedom(binnode *branch) const
{
   if (branch->left==0) // && (branch->right==0))
      return branch->Model.degreesOfFreedom() ; ;
   return branch->Construct.degreesOfFreedom() + 
     degreesOfFreedom(branch->left) + degreesOfFreedom(branch->right) ;
}
