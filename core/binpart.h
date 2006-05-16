#if !defined(BINPART_H)
#define BINPART_H

// ************************************************************
//
//              class    binPartition       
//                       ------------
//
// generates all unique non-empty binary partitions for values 1..N
//
// ************************************************************
class binPartition
{
   marray<int> left ; // left partition table
   int N ; // size
   boolean incLeft(void) ;

public:
  marray<boolean> leftPartition ;
  binPartition(int size) { N=size; left.create(N,0) ; leftPartition.create(N+1) ; }
  long int noPositions() ;
  ~binPartition() { }
  boolean increment() ;
} ;

#endif

