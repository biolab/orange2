#include <stdlib.h>

#include "general.h"
#include "utils.h"
#include "error.h"
#include "contain.h"



// get bootstrap replicates for random forests
void bootstrapSample(int size, marray<int> &data, marray<int> &ib, marray<boolean> &oob) {
	ib.create(size);
 	oob.create(size, TRUE) ;
    int i, sel ;
    // prepare data for the bag
	for (i = 0 ; i < size ; i++) {
       sel = randBetween(0, size) ;
	   ib[i] = data[sel] ;
	   oob[sel] = FALSE ;
	}
}

// get random samples for random forests
void randomSample(int size, double prop, marray<int> &data, marray<int> &ib, marray<boolean> &oob) {
    int ibSize = int(prop * size) ;
	ib.create(ibSize);
 	oob.create(size, TRUE) ;
    int i, sel ;
    // prepare data for the bag
	for (i = 0 ; i < ibSize ; i++) {
	   do {
			sel = randBetween(0, size) ;
	   } while (oob[sel] == FALSE) ;
	   ib[i] = data[sel] ;
	   oob[sel] = FALSE ;
	}
}

// probabilisticaly shuffles the values of valArray, so that every value is changed and 
//the distribution of values is approximately the same 
void shuffleChange(int noValues, marray<int> &valArray) {
   marray<int> distr(noValues+1, 0) ;
   int i, j, value ;
   for (i=0 ; i < valArray.len() ; i++) 
	   ++ distr[valArray[i]] ;
   // change to cumulative distribution
   distr[0] = 0 ;
   for (j=1 ; j <= noValues ; j++) 
	   distr[j] += distr[j-1] ;
   int all = distr[noValues] ;
   for (i=0 ; i < valArray.len() ; i++) {
	   do {
		   value = randBetween(0, all) ;
		   j=1 ;
		   while (value > distr[j])
              j++ ;
		   value = j ;
	   } while (value == valArray[i]) ;
	   valArray[i] = value ;
   }
}
