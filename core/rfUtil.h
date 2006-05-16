#if !defined(RFUTIL_H)
#define RFUTIL_H

void bootstrapSample(int size, marray<int> &data, marray<int> &ib, marray<boolean> &oob) ;
void randomSample(int size, double prop, marray<int> &data, marray<int> &ib, marray<boolean> &oob) ;
void shuffleChange(int noValues, marray<int> &valArray) ;


#endif
