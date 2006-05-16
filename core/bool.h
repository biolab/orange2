#ifndef _Boolean_
#define _Boolean_

typedef bool    Boolean;

const   False = false;
const   True  = true;

/*
   #if defined(__MSDOS__) && defined(_Windows) && defined(__BORLANDC__)
      #define TRUE    1
      #define FALSE   0
      typedef int     BOOL; 
   #else
      // Boolean 
      #ifndef _BooleanType  // Boolean from VAC++
        const   False = 0;
        const   True  = 1;
        typedef int     Boolean;
      #endif
      // BOOL 
      #ifdef TRUE 
         #undef TRUE 
      #endif
      #define TRUE    True
      #ifdef FALSE 
         #undef FALSE
      #endif
      #define FALSE   False
      #ifdef BOOL
         #undef BOOL 
      #endif
      #define BOOL    Boolean
   #endif
*/
#endif
