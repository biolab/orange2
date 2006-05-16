#if !defined(CONTAIN_H)
#define CONTAIN_H

#include <stdlib.h>
#include "general.h"
#include "error.h"



template <class T> T Mmin(T a, T b) {  return ((a<b) ? a : b ) ; }
template <class T> T Mmax(T a, T b) {  return ((a>b) ? a : b ) ; }
template <class T>  void swap(T &X, T &Y) { T temp = X ; X = Y ; Y = temp ; }


/********************************************************************
*
*   Name:                class marray
*
*   Description: onedimensional dinamic array of type T
*
*********************************************************************/

template<class T>
class marray
{
    int size, edge ;
    T *table ;

public:

    marray() { size = edge = 0 ; table = 0 ; }
    marray(int a) ;
    marray(int a, T InitValue) ;
    marray(marray<T> &Source) ;
    // marray(int sourceSize, T *Source) ;
    marray<T>& operator= (const marray<T> &Source) ; 
    void copy(const marray<T> &Source) ;
    ~marray() { delete [] table ; table = 0 ;}
    boolean defined(void)  ;
    void create(int a) ;
    void create(int a, T InitValue) ;
    inline T& operator[] (int a)
      {
         #if defined(DEBUG)
            if ( a>=size || a<0)
               error("marray, operator []:","bounds check failed !") ;
         #endif
         return table[a] ;
      }
    inline T const & operator[] (int a) const
      {
         #if defined(DEBUG)
            if ( a>=size || a<0)
               error("marray, const operator []:","bounds check failed !") ;
         #endif
         return table[a] ;
      }
   boolean member(T &X) const ;
   int memberPlace(T &X) const ;
   int lessEqPlace(T &X) ;
   void addEnd(T& X) ;
   int addToAscSorted(T& X) ; 
   inline void incEdge(void)
      {
         edge++;
         #if defined(DEBUG)
            if (edge > size)
               error("marray::incEdge:", "adding past the edge of a table.") ;
         #endif
      }
   inline void decEdge(void)
      {
         edge--;
         #if defined(DEBUG)
            if (edge < 0)
               error("marray::decEdge:", "decrementing edge of empty table") ;
         #endif
      }
   void clear(void) { edge=0 ; }
   void setFilled(int filled) ;
   inline int filled(void) { return edge; }
   inline int len(void) { return size ; }
   void init(T InitValue) ;
   void init(int From, int To,T InitValue) ;
   void sort(int(*compare)(const void *element1, const void *element2))
       { qsort((void *)table, edge, sizeof(T), compare); }
   void sort(int fromIdx, int toIdx, int (*compare)(const void *element1, const void *element2)) ;
   void qsortAsc(void) ;
   void qsortDsc(void) ;
   void sortKasc(int K) ;
   void sortKdsc(int K) ;
   void pushdownAsc(int first, int last) ;
   void pushdownDsc(int first, int last) ;
   T& select(int k) ;  // selects k-th element in the filled part of array
   void addPQmin(T &X) ;
   void addPQmax(T &X) ;
   void deleteMinPQmin(T &X) ;
   void deleteMaxPQmax(T &X) ;

   void destroy(void) {  delete [] table ; table = 0; size = 0 ; edge= 0 ;}
   void enlarge(int newSize) ;

   int operator== (marray<T> &Y) ;
   int operator< (marray<T> &) { return 0 ;} ;
   int operator> (marray<T> &) { return 0 ;} ;

   T* toArray(void) { return table ; }
   void wrap(int sourceSize, T* source) ;
   T* unWrap(int &outSize)  ;
   void shuffle(void) ;
}  ;




// constructor which creates array of size a 
template<class T> marray<T>::marray(int a)
{
   table=0 ;
   create(a) ;
}

// constructor which creates array of size a and initializes it to InitValue
template<class T> marray<T>::marray(int a, T InitValue)
{
    table = 0 ;
    create(a, InitValue) ;
}



// copy constructor
template<class T> marray<T>::marray(marray<T> &Source) 
{
   size = edge = 0 ;
   table = 0 ;
   copy(Source) ;
 }


//template<class T> marray<T>::marray(int sourceSize, T *Source) 
//{
//#if defined(DEBUG)
//    if (sourceSize <= 0 || Source == 0)
//      error("marray::maray", "source table is not appropriate.") ;
//#endif
//
//   size = sourceSize ;
//   edge = 0 ;
//   table = new T[size] ;
//   for (int i=0 ; i < size ; i++)
//       table[i] = Source[i] ;
//}

// assignment operator
template<class T> marray<T>& marray<T>::operator= (const marray<T> &Source) 
{
   copy(Source) ;
   return *this ;
}

// copies the Source array
template<class T> void marray<T>::copy(const marray<T> &Source)
{
    if (&Source == this)
        return ;

    if (Source.table)
    {
       create(Source.size) ;
       edge = Source.edge ;
       for (int i=0 ; i <Source.size; i++)
         table[i] = Source.table[i] ;
    }
    else
       destroy() ;
}


// is this table defined at all
template<class T> boolean marray<T>::defined(void)
{
   if (table)
      return TRUE ;
   else
      return FALSE ;
}



// creates array of size a 
template<class T> void marray<T>::create(int a)
{
   
	if (table)
      delete [] table ;
  
   size = a ;
   edge = 0 ;

#if defined (DEBUG)
   if (size < 0)
      error("marray::create", "size of a table is less than zero.") ;
#endif

   if (size>0)   
      table = new T[size];
    else
		table = 0 ;

#if defined (DEBUG)
   if (table == 0 && size > 0)
      error("marray::create", "cannot allocate memory.") ;
#endif
}



// creates array of size a and initializes it to InitValue
template<class T> void marray<T>::create(int a, T InitValue)
{
   create(a) ;
   init(InitValue) ;
}



// is the element X member of the filled array
template<class T> boolean marray<T>::member(T &X) const
{
   for (int i=0 ; i < edge ; i++)
   {
      if (table[i] == X)
         return TRUE ;
   }
   return FALSE ;
}



// returns index of the element X if it exists in the filled table
template<class T> int marray<T>::memberPlace(T &X) const
{
   for (int i=0 ; i < edge ; i++)
   {
      if (table[i] == X)
         return i ;
   }
   return -1 ;
}

// returns index of the element X is immediately less or equal s in the filled table
template<class T> int marray<T>::lessEqPlace(T &X) 
{
   for (int i=0 ; i < edge ; i++)
   {
      if (! (table[i] > X) )
         return i ;
   }
   return edge ;
}


// adds X to nthe end of filled array
template<class T> void marray<T>::addEnd(T& X)
{
   #if defined(DEBUG)
      if (edge >= size)
         error("marray::addEnd:", "Adding beyond the size of a table.") ;
   #endif
   table[edge++] = X ;
}



// returns the place to which the X was inserted
template<class T> int marray<T>::addToAscSorted(T& X)
{
   #if defined(DEBUG)
      if (edge >= size)
         error("marray::addToAscSorted:", "Adding beyond the size of a table.") ;
   #endif
   // find the appropriate place with binary search (bisection)
   int lower = 0 ;
   int upper =edge  ; 
   int middle ;
   while (lower < upper)
   {
      middle = (lower+upper)/2 ;
      if (! (X < table[middle])) // >=
         lower = middle+1 ;
      else
         upper = middle ;
   }
   // shift larger
   for (int i=edge ; i > upper ; i--)
      table[i] = table[i-1] ;
   // set value and edge
   table[upper] = X ;
   edge++ ;
   return upper ;
}



// sets the point to which array is filled
template<class T> void marray<T>::setFilled(int filled)
{
   #if defined(DEBUG)
      if (filled > size)
         error("marray::setFilled:","moving edge beyond size.") ;
   #endif
   edge = filled ;
}



// initializes the array to the value InitValue
template<class T> void marray<T>::init(T InitValue)
{
    for (int i=0 ; i<size ; i++)
       table[i] = InitValue ;
}



// initializes the array from From to To to the value of InitValue
template<class T> void marray<T>::init(int From, int To, T InitValue)
{
   #if defined(DEBUG)
      if (To > size)
         error("marray::initFromTo", "Initializing over the upper bound of the table.") ;
      if (From < 0)
         error("marray::initFromTo", "Initializing negative indexes of the table.") ;
   #endif

    for (int i=From ; i<To ; i++)
       table[i] = InitValue ;
}



// enlarges the array to newSize and preserves the old contens
template<class T> void marray<T>::enlarge(int newSize)
{
   if (newSize <= size)
      return ;

   T* newTable = new T[newSize] ;
   #if defined(DEBUG)
     if (newTable == 0)
       error("marray::enlarge :","cannot allocate enough memory") ;
   #endif
   for (int i=0 ; i < size ; i++)
      newTable[i] = table[i] ;

   delete [] table ;
   table = newTable ;
   size = newSize ;
}


template<class T>void marray<T>::wrap(int sourceSize, T* source) 
{
    destroy() ;
    size = sourceSize ;
    edge=0 ;
    table=source ;
}

template<class T>T* marray<T>::unWrap(int &outSize) 
{
    outSize = size ;
    size = 0 ;
    T* retTable = table ;
    table = 0 ;
    return retTable ;
}

// operator of equality
template<class T> int marray<T>::operator== (marray<T> &Y)
{
  if  ( len() != Y.len() || filled() != Y.filled() ) 
     return 0 ;
  for (int i=0 ; i < filled() ; i++)
    if (! (table[i] == Y[i]) )
      return 0 ;
  return 1 ;
}



// sorts table from to Idx with given comparison function
template<class T> void marray<T>::sort(int fromIdx, int toIdx, int (*compare)(const void *element1, const void *element2)) 
{
#if defined(DEBUG)
   if (fromIdx < 0 || toIdx > size || fromIdx > toIdx)
          error("marray::sort/3", "incorrect indexes") ;
#endif
    qsort((void *)(table + fromIdx), toIdx - fromIdx, sizeof(T), compare);
}




// the resuting table has the elements
// table[edge-1-K] up to table[edge-1] sorted
// ascending - the biggest value being table[edge-1]
//               -----------
// so we actually find K biggest numbers
template<class T> void marray<T>::sortKasc(int K)
{

    #if defined(DEBUG)
    if (K < 0 ) 
       error("marray::sortKdsc", "the demanded number of sorted items is out of range") ;
    #endif 

	if (K > edge)
		K=edge ;

    // finding K biggest in increasing order with heapsort method

   // initial establishment of partially ordered tree
   int i ;
   for (i = edge/2 ; i >= 1 ; i--)
     pushdownAsc(i, edge) ;

   // main loop
   int lower = Mmax(edge - K, 1) ;
   i = edge ;
   while (i > lower)
   {
      i-- ;
      swap(table[i], table[0]) ;
      pushdownAsc(1, i) ;
   }
   // here elements are ordered in descending order from 
   // lower to edge
}



// the resuting table has the elements
// table[edge-1-K] up to table[edge-1] sorted
// descending - the smallest value being table[edge-1]
//               -----------
// so we actually find K smallest numbers
template<class T> void marray<T>::sortKdsc(int K)
{

    #if defined(DEBUG)
    if (K < 0 ) 
       error("marray::sortKdsc", "the demanded number of sorted items is out of range") ;
    #endif 

	if (K > edge)
		K=edge ;

    // finding K biggest in increasing order with heapsort method

   // initial establishment of partially ordered tree
   int i ;
   for (i = edge/2 ; i >= 1 ; i--)
     pushdownDsc(i, edge) ;

   // main loop
   int lower = Mmax(edge - K, 1) ;
   i = edge ;
   while (i > lower)
   {
      i-- ;
      swap(table[i], table[0]) ;
      pushdownDsc(1, i) ;
   }
   // here elements are ordered in descending order from 
   // lower to edge
}




// ***************************************************************************
//
//                        pushdownAsc
//               pushes elements down the heap and restores POT property
//
// assumes table[first], ... , table[last]obeys partially ordered tree 
// property
// except possibly for the children of table[first]. The procedure pushes
// table[first] down until the partially ordered tree property is restored
//
// ***************************************************************************
template<class T> void marray<T>::pushdownAsc(int first, int last)
{
   int r = first ;  // the current position of table[first]
   int child1, child2, parent ;

   int limit = last / 2 ;
   while ( r <= limit )
   {
       child2 = 2 * r ;
       child1 = child2 - 1 ;
       parent = r - 1 ;
       if (last == child2 ) // r has one child at 2*r
       {
          if ( table[parent] < table[child1] )
             swap(table[parent], table[child1]) ;
          break ; // forces a break from the while-loop
       }
       else  // r has two children at 2*r and 2*r+1
          if ( table[parent] < table[child1]  &&
               ! (table[child1] < table[child2]) )
          {  // swap r with left children
             swap(table[parent], table[child1]) ;
             r = child2 ;
          }
          else
            if ( table[parent] < table[child2]  &&
                 table[child1] < table[child2] )
            {  // swap r with left children
               swap(table[parent], table[child2]) ;
               r = child2 + 1 ;
            }
            else // r does not violate partially ordered tree property
              break ;
   }

}


// ***************************************************************************
//
//                        pushdownDsc
//               pushes elements down the heap and restores POT property
//
// assumes table[first], ... , table[last]obeys partially ordered tree 
// property
// except possibly for the children of table[first]. The procedure pushes
// table[first] down until the partially ordered tree property is restored
//
// ***************************************************************************
template<class T> void marray<T>::pushdownDsc(int first, int last)
{
   int r = first ;  // the current position of table[first]
   int child1, child2, parent ;

   int limit = last / 2 ;
   while ( r <= limit )
   {
       child2 = 2 * r ;
       child1 = child2 - 1 ;
       parent = r - 1 ;
       if (last == child2 ) // r has one child at 2*r
       {
          if ( table[parent] > table[child1] )
             swap(table[parent], table[child1]) ;
          break ; // forces a break from the while-loop
       }
       else  // r has two children at 2*r and 2*r+1
          if ( table[parent] > table[child1]  &&
               ! (table[child1] > table[child2]) )
          {  // swap r with left children
             swap(table[parent], table[child1]) ;
             r = child2 ;
          }
          else
            if ( table[parent] > table[child2]  &&
                 table[child1] > table[child2] )
            {  // swap r with left children
               swap(table[parent], table[child2]) ;
               r = child2 + 1 ;
            }
            else // r does not violate partially ordered tree property
              break ;
   }

}



// ***************************************************************************
//
//                        qsortAsc
//            sorts elements with Quicksort in ascending order
//       uses a variant of median of three for selecting partitioning element a
//
// ***************************************************************************
template<class T> void marray<T>::qsortAsc(void)
{
	int i, ir=edge-1,j,k,l=0 ;
	T a;
   const int smallArraySize = 7 ;
   const int maxStack = 100 ; // 2*log_2 N i.e., enough for 2^50 elements

	marray<int> StackIdx(maxStack);  // stack for indexes
   int idxStack=-1 ; // shows last used

	for (;;) 
   {
		if (ir-l < smallArraySize) // insertion sort if small enough
      {
			for (j=l+1 ; j<=ir ; j++) 
         {
				a=table[j];
				for (i=j-1;i>=0;i--) 
            {
					if (! (table[i] > a) ) 
                  break;
					table[i+1]=table[i];
				}
				table[i+1]=a;
			}
			if (idxStack < 0) 
            break;
         // pop new boundaries from stack and start again
			ir=StackIdx[idxStack--];   
			l=StackIdx[idxStack--];
		} 
      else {
			k=(l+ir) / 2 ;
			// median of three: first, middle and last
         swap(table[k],table[l+1]) ;
			if (table[l+1] > table[ir]) 
         	swap(table[l+1], table[ir]) ;
			if (table[l] > table[ir]) 
				swap(table[l],table[ir]) ;
			if (table[l+1] > table[l]) 
				swap(table[l+1],table[l]) ;
         // table[l] now contain median
			i=l+1;
			j=ir;
			a=table[l];
			// partition the elements
         for (;;) 
         {
				do i++; while (table[i] < a);
				do j--; while (table[j] > a);
				if (j < i) 
               break;
				swap(table[i],table[j]);
			}
			table[l]=table[j];
			table[j]=a;
			#if defined(DEBUG)
           if (idxStack+2 >= maxStack) 
             error("marray::qsortAsc","maxStack too small in sort");
         #endif 
			if (ir-i+1 >= j-l) 
         {
				StackIdx[++idxStack]=i;
				StackIdx[++idxStack]=ir;
				ir=j-1;
			} 
         else {
				StackIdx[++idxStack]=l;
				StackIdx[++idxStack]=j-1;
				l=i;
			}
		}
	}
}


// ***************************************************************************
//
//                        qsortDsc
//            sorts elements with Quicksort in descending order
//       uses a variant of median of three for selecting partitioning element a
//
// ***************************************************************************
template<class T> void marray<T>::qsortDsc(void)
{
	int i, ir=edge-1,j,k,l=0 ;
	T a;
   const int smallArraySize = 7 ;
   const int maxStack = 100 ; // 2*log_2 N i.e., enough for 2^50 elements

	marray<int> StackIdx(maxStack);  // stack for indexes
   int idxStack=-1 ; // shows last used

	for (;;) 
   {
		if (ir-l < smallArraySize) // insertion sort if small enough
      {
			for (j=l+1 ; j<=ir ; j++) 
         {
				a=table[j];
				for (i=j-1;i>=0;i--) 
            {
					if (! (table[i] < a) ) 
                  break;
					table[i+1]=table[i];
				}
				table[i+1]=a;			}
			if (idxStack < 0) 
            break;
         // pop new boundaries from stack and start again
			ir=StackIdx[idxStack--];   
			l=StackIdx[idxStack--];
		} 
      else {
			k=(l+ir) / 2 ;
			// median of three: first, middle and last
         swap(table[k],table[l+1]) ;
			if (table[l+1] < table[ir]) 
         	swap(table[l+1], table[ir]) ;
			if (table[l] < table[ir]) 
				swap(table[l],table[ir]) ;
			if (table[l+1] < table[l]) 
				swap(table[l+1],table[l]) ;
         // table[l] now contain median
			i=l+1;
			j=ir;
			a=table[l];
			// partition the elements
         for (;;) 
         {
				do i++; while (table[i] > a);
				do j--; while (table[j] < a);
				if (j < i) 
               break;
				swap(table[i],table[j]);
			}
			table[l]=table[j];
			table[j]=a;
			#if defined(DEBUG)
           if (idxStack+2 >= maxStack) 
             error("marray::qsortDsc","maxStack too small in sort");
         #endif 
			if (ir-i+1 >= j-l) 
         {
				StackIdx[++idxStack]=i;
				StackIdx[++idxStack]=ir;
				ir=j-1;
			} 
         else {
				StackIdx[++idxStack]=l;
				StackIdx[++idxStack]=j-1;
				l=i;
			}
		}
	}
}




// ***************************************************************************
//
//                        select
//     selects k-th element in the filled part of the array with side effect of
//     rearanging the elements
//
// ***************************************************************************
template<class T> T& marray<T>::select(int k) 
{
	int i,j,left,right,mid;

#if defined(DEBUG)
    if (k < 0 || k > edge)
        error("marray::select", "k is out of range") ;
#endif
	left=0;
	right=edge-1;
	while (TRUE)
    {
		if (right <= left+1) 
        {
			if (right == left+1 && table[right] < table[left]) // two elements
            {
				swap(table[left], table[right]) ;
			}
			return table[k];
		} 
        else 
        {
            // choose median of left, middle and right element as partitioning element part
            // Also rearange so that table[left+1] <= table[left], table[right] >= table[left]
			mid=(left+right) / 2;     

			swap(table[mid], table[left+1]) ;
			if (table[left+1] > table[right]) 
				swap(table[left+1], table[right]) ;
			if (table[left] > table[right]) 
				swap(table[left], table[right]) ;
			if (table[left+1] > table[left]) 
				swap(table[left+1], table[left]) ;
			
            // initialize pointers for partitioning
			i=left+1;
			j=right;
			//part=table[left];  partitioning element is table[left]
			while (TRUE) 
            {
				do i++; while (table[i] < table[left]);  // scan up to find element > part
				do j--; while (table[j] > table[left]);  // scan down to find element < part
				if (j < i)  // pointers crossed, partitioning complete
                   break;
				swap(table[i], table[j]); 
			}
			swap(table[left], table[j]) ;  // insert partitioning element
			// keep active the partition that contains the k-th element
            if (j >= k) 
                right=j-1;  
			if (j <= k) 
                left=i;
		}
	}
   // this point should never be reached
   return table[-1] ; // for the sake of too 'smart' compilers
}



// adding to heap where the top element (0th) element is the smallest
template<class T> void marray<T>::addPQmin(T &X) 
{
   #if defined(DEBUG)
    if (edge >= size)
         error("marray::addPQmin", "Adding beyond the size of a PQtable.") ;
   #endif
   int newPos = edge ;
   edge++ ;
   int parent  = (newPos+1)/2 -1 ;
   while (parent >= 0 && table[parent] > X)
   {
      table[newPos] = table[parent] ;
      newPos = parent ;
      parent = (parent+1)/2 -1 ;
   }
   // new element is added when its position is known
   table[newPos] = X ;
}


// adding to heap where the top element (0th) element is the smallest
template<class T> void marray<T>::addPQmax(T &X) 
{
   #if defined(DEBUG)
    if (edge >= size)
         error("marray::addPQmax", "Adding beyond the size of a PQtable.") ;
   #endif
   int newPos = edge ;
   edge++ ;
   int parent  = (newPos+1)/2 -1 ;
   while (parent >= 0 && table[parent] < X)
   {
      table[newPos] = table[parent] ;
      newPos = parent ;
      parent = (parent+1)/2 -1 ;
   }
   // new element is added when its position is known
   table[newPos] = X ;
}


template<class T> void marray<T>::deleteMinPQmin(T &X) 
{
   #if defined(DEBUG)
    if (edge <= 0)
         error("marray::deleteMinPQmin", "The PQtable is empty.") ;
   #endif
    
   X = table[0] ;
   edge-- ;
   table[0] = table[edge] ;
   pushdownDsc(1,edge) ;
}   



template<class T> void marray<T>::deleteMaxPQmax(T &X) 
{
  #if defined(DEBUG)
    if (edge <= 0)
         error("marray::deleteMaxPQmax", "The PQtable is empty.") ;
   #endif
    
   X = table[0] ;
   edge-- ;
   table[0] = table[edge] ;
   pushdownAsc(1,edge) ;
}   

int randBetween(int from, int to) ;

template<class T> void marray<T>::shuffle(void) {
	for (int i=edge-1 ; i > 0 ; i--) 
		swap(table[i], table[randBetween(0, i+1)]) ;
}



/********************************************************************
*
*   Name:                class mmatrix
*
*   Description: twodimensional dinamic array
*                we suppose that dim2 > dim1 so to spare
*                some space for pointers, we invert dimensions
*
*********************************************************************/
template<class Type>
class mmatrix
{
//typedef Type* PType ;

   int dim1,dim2;
   Type **table;

 public:

    mmatrix(int a,int b);
    mmatrix(int a,int b, Type Value);
    mmatrix();
    ~mmatrix();
    void create(int a, int b) ;
    void create(int a, int b, Type Value) ;
    void destroy(void);
    mmatrix<Type>& operator= (const mmatrix<Type> &Source) ; 
    void copy(const mmatrix<Type> &Source) ;
    boolean defined(void)  ;
    inline Type& operator() (int a,int b) const
      {
         #if defined(DEBUG)
            if (b>=dim2 || b < 0)
               error("mmatrix::operator() :","2nd dimension violation.") ;
            if (a>=dim1 || a < 0)
               error("mmatrix::operator() :","1st dimension violation.") ;
         #endif
         return (table[b])[a] ;
      }
    inline void Set(int a, int b, Type Value)
      {
         #if defined(DEBUG)
            if (b>=dim2 || b < 0)
               error("mmatrix::Set :","2nd dimension violation.") ;
            if (a>=dim1 || a < 0)
               error("mmatrix::Set :","1st dimension violation.") ;
         #endif
         table[b][a] = Value ;
      }
    void swallow(mmatrix<Type>& target); // destroys target
    void init(Type Value) ;
    void addColumns(int newDim2) ;
    void changeColumns(int First, int Second) ;
    boolean equalColumns(int First, int Second) ;
    void copyColumn(int Source, int Target) ;
	void outColumn(int source, marray<Type> &target) ;
	void inColumn(marray<Type> &source, int target) ;
    Type** toArray(void) { return table ; }

};



// basic constructor
template<class Type> mmatrix<Type>::mmatrix()
{
   dim1 = dim2 = 0 ;
   table = 0;
}



//  constructor with dimensions
template<class Type> mmatrix<Type>::mmatrix(int a,int b)
{
   table=0 ;
   create(a,b) ;
}



//  constructor with dimensions and initialization
template<class Type> mmatrix<Type>::mmatrix(int a,int b, Type Value)
{
   table = 0 ;
   create(a,b,Value) ;
}



// destructor
template<class Type> mmatrix<Type>::~mmatrix()
{
   destroy();
}


//************************************************************
//
//                        mmatrix::destroy
//                        ---------------
//
//                releases data reserved on free store
//
//************************************************************
template<class Type> void mmatrix<Type>::destroy()
{
  if (table)
  {
     for  (int i=0 ; i<dim2 ; i++)
     {
        if (table[i])
          delete [] table[i] ;
        table[i] = 0 ;
     }
     delete [] table ;
     table = 0 ;
  }
  dim1 = dim2 = 0 ;
}





//  creation of matrix
template<class Type> void mmatrix<Type>::create(int a,int b)
{
   destroy() ;
   dim1 = a ;
   dim2 = b ;
   #if defined(DEBUG)
     if (dim1 < 0)
        error("mmatrix::create :","1st dimension of array is negative") ;
     if (dim2 < 0)
        error("mmatrix::create :","2nd dimension of array is negative") ;
   #endif

   table = new Type*[dim2] ;

   #if defined(DEBUG)
     if (table==0 && dim2 > 0)
        error("mmatrix::create :","could not obtain enough memory.") ;
   #endif

   for (int i=0 ; i < dim2 ; i++)
   {
      table[i] = new Type[dim1] ;

      #if defined(DEBUG)
       if (table[i]==0 && dim1 > 0)
          error("mmatrix::create :","could not allocate enough memory.") ;
      #endif
   }
}


//  creation of matrix
template<class Type> void mmatrix<Type>::create(int a,int b, Type Value)
{
   create(a,b) ;
   init(Value) ;
}


// assignment operator
template<class Type> mmatrix<Type>& mmatrix<Type>::operator= (const mmatrix<Type> &Source) 
{
   copy(Source) ;
   return *this ;
}

// copies the Source mmatrix
template<class Type> void mmatrix<Type>::copy(const mmatrix<Type> &Source)
{
    if (&Source == this)
        return ;

    if (Source.table)
    {
       create(Source.dim1, Source.dim2) ;
       int i, j ;
       for (i=0 ; i <dim2; i++)
          for (j=0 ; j < dim1 ; j++)
             table[i][j] = Source.table[i][j] ;
    }
    else
       destroy() ;
}




// is the matrix defined
template<class Type> boolean mmatrix<Type>::defined(void)
{
   if (table)
      return TRUE ;
   else
      return FALSE ;
}




//************************************************************
//
//                        mmatrix::swallow
//                        ------------------
//
//       assignment operator:
//          we have to destroy old dinamical structure, create
//          a new one and move the data
//                        ----
//************************************************************
template<class Type> void mmatrix<Type>::swallow(mmatrix<Type>& copy)
{
   destroy();
   dim1=copy.dim1;
   dim2=copy.dim2;
   table = copy.table ;
   copy.table = 0 ;
}

template<class Type> void mmatrix<Type>::init(Type Value)
{
    int i,j ;
    for (i=0 ; i < dim2 ; i++)
      for (j=0 ; j < dim1 ; j++)
         table[i][j] = Value ;
}


template<class Type> void mmatrix<Type>::addColumns(int newDim2)
{
   if (newDim2 <= dim2)
      return ;
   Type **tableNew ;
   tableNew = new Type*[newDim2] ;
   #if defined(DEBUG)
     if (tableNew==0)
        error("mmatrix::addColumns :", " cannot allocate enough memory") ;
   #endif

   int i ;
   for (i=0; i < dim2 ; i++)
     tableNew[i] = table[i] ;

   delete [] table ;
   table = tableNew ;

   for (i=dim2 ; i < newDim2 ; i++)
   {
      table[i] = new Type[dim1] ;
      #if defined(DEBUG)
        if (table[i]==0 && dim1 > 0)
          error("mmatrix::addColumns :", " cannot allocate memory for the columns") ;
      #endif
   }

   dim2 = newDim2 ;
}

template<class Type> void mmatrix<Type>::changeColumns(int First, int Second)
{
   #if defined(DEBUG)
     if (First >= dim2 || Second >= dim2 || First < 0 || Second < 0)
       error("mmatrix::changeColumns :","change indexes out of range") ;
   #endif
   Type *temp ;
   temp = table[First] ;
   table[First] = table[Second] ;
   table[Second] = temp ;
}


template<class Type> boolean mmatrix<Type>::equalColumns(int First, int Second)
{
   #if defined(DEBUG)
     if (First >= dim2 || Second >= dim2 || First < 0 || Second < 0)
       error("mmatrix::changeColumns :","change indexes out of range") ;
   #endif
   for (int i=0 ; i < dim1 ; i++)
      if (! (table[First][i] == table[Second][i]) )
           return FALSE ;
    return TRUE ;
}

template<class Type> void mmatrix<Type>::copyColumn(int Source, int Target)
{
   #if defined(DEBUG)
     if (Source >= dim2 || Target >= dim2 || Source < 0 || Target < 0)
       error("mmatrix::copyColumn :","indexes out of range") ;
   #endif
   for (int i=0 ; i < dim1 ; i++)
      table[Target][i] = table[Source][i] ;
}

template<class Type> void mmatrix<Type>::outColumn(int source, marray<Type> &target) {
   #if defined(DEBUG)
     if (target.len() < dim1 || source < 0 || source >= dim2 )
       error("mmatrix::outColumn :","incompatible sizes or indexes out of range") ;
   #endif
   for (int i=0 ; i < dim1 ; i++)
      target[i] = table[source][i] ;
}

template<class Type> void mmatrix<Type>::inColumn(marray<Type> &source, int target)  {
   #if defined(DEBUG)
     if (source.filled() != dim1 || target < 0 || target >= dim2 )
       error("mmatrix::inColumn :","incompatible sizes or indexes out of range") ;
   #endif
   for (int i=0 ; i < dim1 ; i++)
     table[target][i] = source[i] ;
}


template<class T> void insertToSortedAsc(T& X , marray<T> &Array)
{
   #if defined(DEBUG)
      if (Array.filled() >= Array.len())
         error("insertToSortedAsc:", "Inserting into the full table.") ;
   #endif
   int place = 0 ;
   while (place < Array.filled() && Array[place] <= X)
     place++ ;
   for (int i = Array.filled() ; i > place; i-- )
      Array[i] = Array[i-1] ;
   Array[place] = X ;
   Array.incEdge()  ;
}

template<class T> void insertToSortedDesc(T& X , marray<T> &Array)
{
   #if defined(DEBUG)
      if (Array.filled() >= Array.len())
         error("insertToSortedDesc:", "Inserting into the full table.") ;
   #endif
   int place = 0 ;
   while (place < Array.filled() && Array[place] >= X)
     place++ ;
   for (int i = Array.filled() ; i > place; i-- )
      Array[i] = Array[i-1] ;
   Array[place] = X ;
   Array.incEdge()  ;
}

template<class T> void deleteFromSortedAsc(T& X , marray<T> &Array)
{
   int place = 0 ;
   // deleting last occurence
   while (place < Array.filled() && Array[place] <= X)
     place++ ;
   #if defined(DEBUG)
       if (place > Array.filled() || Array[place-1] != X)
          error("deleteFromSortedAsc:","deleting unexisting value") ;
   #endif
   while (place < Array.filled())
   {
      Array[place-1] = Array[place] ;
      place++ ;
   }
  Array.decEdge()  ;
}

template<class T> void deleteFromSortedDesc(T& X , marray<T> &Array)
{
   int place = 0 ;
   // deleting last occurence
   while (place < Array.filled() && Array[place] >= X)
     place++ ;
   #if defined(DEBUG)
       if (place > Array.filled() || Array[place-1] != X)
          error("deleteFromSortedAsc:","deleting unexisting value") ;
   #endif
   while (place < Array.filled())
   {
      Array[place-1] = Array[place] ;
      place++ ;
   }
  Array.decEdge()  ;
}




#endif
