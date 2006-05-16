#include "contain.h"
#include "utils.h"
#include "binpart.h"


boolean binPartition::incLeft(void)
{
  // initial increment
  if (left[0] == 0)
  {
    left[0] = 1 ;
    return TRUE ;
  }

  int i, position = 0 ;
  while (TRUE)
  {
    // have we reached the sentinel
    if (left[position] == 1)
    {
      if (position == N-2) // is this the last position
         return FALSE ;
      // otherwise shift right
      position ++ ;
      left[position] = 0 ; // we increment to 1 in the next sentence
    }
    if (left[position] < N - position) // is there still room for increment
    {
      left[position] ++ ;
      for (i=position - 1 ; i >= 0 ; i--) // set also others
        left[i] = left[i+1] + 1 ;
      return TRUE ;
    }
    position ++ ;
  }
  return FALSE ;
}



boolean binPartition::increment(void)
{
   if (incLeft() )
   {
     // due to readability reasons we will return smaller partition
     int filled = 0 ;
     while (left[filled] != 1)
       filled ++ ;
     boolean selected = TRUE ;
     boolean reversed = FALSE ;
     if (filled +1 > N/2)
     {
       selected = FALSE ;
       reversed = TRUE ;
     }
     
     // set values in partition
     leftPartition.init(reversed) ;
     while (filled >= 0)
     {
       leftPartition[left[filled]] = selected ;
       filled --  ;
     }

     return TRUE ;
   }
   else 
     return FALSE ;
}


long int binPartition::noPositions(void)
{
   // estimate for number of positions
   long int number = 0 ;
   int i  ;
   for (i=1 ; i <= (N-1)/2 ; i++)
     number += binom(N,i) ;
   
   if (N % 2 == 0) // even N
     number += binom(N, i)/2 ;
  
   return number ;
}

