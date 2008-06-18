#ifndef cMersenneTwister_h_INCLUDED
#define cMersenneTwister_h_INCLUDED

// This is the ``Mersenne Twister'' random number generator MT19937, 
// which generates pseudorandom integers uniformly distributed in 
// 0..(2^32 - 1) starting from any odd seed in 0..(2^32 - 1) with
// period 2^19937 -1 with 623-dimmensional equidistibution up to
// 32-bits of accurancy.
// This is Shaw Cookus Implementation of algorithm invented by
// Takuji Nishimura embedded into C++ class by me, that is
// Maciek Urbañski.
// Amortized cost of generating 32-bit integer is 33 CPU cycles on
// my K6-2 450MHz (MSVC 6.0/SP4). It's fast. ;-)
// For detailed properties - there is a paper: "Mersenne Twister:
// A 623-dimmensionally equidstributed uniform pseudorandom number
// generator" by Makoto Matsumoto and Takuji Nishimura.

#define N               (624)
#define M               (397)
#define K               (0x9908B0DFU)
#define hiBit(u)        ((u) & 0x80000000U)
#define loBit(u)        ((u) & 0x00000001U)
#define loBits(u)       ((u) & 0x7FFFFFFFU)
#define mixBits(u, v)   (hiBit(u)|loBits(v))
#define SEED0           (4357U)


class cMersenneTwister
{
// Made everything public by JD (need it for pickling)
public:

  unsigned long   state[N+1];   // state vector
  unsigned long   *next;        // next random
  int      left;                // how many values left


  ////////////////////////////////////////////////////////////////////
  // initialize MT via linear conguential generator 
  // x[n+1] = (69069 * x[n]) mod 2^32
  void Init( unsigned long seed       //32-bit seed
           )
  {
    register unsigned long x = (seed | 1U) & 0xFFFFFFFFU, *s = state;
    register int j;
    for(left = 0, *s++ = x, j = N; --j; *s++ = (x*=69069U) & 0xFFFFFFFFU);
  }
  ////////////////////////////////////////////////////////////////////
  // load state vector 
  void Load( unsigned long *state     //pointer to state (624*4 bytes)
           )
  {
    register unsigned long j, *s = state;
    for( j=N; --j; *s++ = *state++ );
  }
  ////////////////////////////////////////////////////////////////////
  // save state vector 
  void Save( unsigned long *state     //pointer to state (624*4 bytes)
           )
  {
    register unsigned long j, *s = state;
    for( j=N; --j; *state++ = *s++ );
  }
  ////////////////////////////////////////////////////////////////////
  // create MT object
  cMersenneTwister( void )
  {
    left = -1;
  }
  ////////////////////////////////////////////////////////////////////
  // create MT object & init it
  cMersenneTwister( unsigned long seed )
  {
    Init( seed );
  }
  ////////////////////////////////////////////////////////////////////
  // create MT object & load seed
  cMersenneTwister( unsigned long *state )
  {
    Load( state );
  }
  ////////////////////////////////////////////////////////////////////
  // calculates next values 
  unsigned long Reload( void )
  {
    register unsigned long *p0 = state, *p2 = state+2, *pM = state+M, s0, s1;
    register int j;
    if(left < -1) Init( SEED0 );
    left = N-1, next = state+1;
    for(s0 = state[0], s1 = state[1], j = N-M+1; --j; s0 = s1, s1 = *p2++ )
      *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    for(pM = state, j = M; --j; s0 = s1, s1 = *p2++)
      *p0++ = *pM++ ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    s1 = state[0], *p0 = *pM ^ (mixBits(s0, s1) >> 1) ^ (loBit(s1) ? K : 0U);
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9D2C5680U;
    s1 ^= (s1 << 15) & 0xEFC60000U; 
    return ( s1 ^ (s1 >> 18) ); 
  }
  ////////////////////////////////////////////////////////////////////
  // get next value 
  inline unsigned long Random( void )
  {
    register unsigned long y;
    if(--left < 0) return Reload();
    y  = *next++;
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    return(y ^ (y >> 18));
  }
};

#undef N
#undef M
#undef K
#undef hiBit
#undef loBit
#undef loBits
#undef mixBits
#undef SEED0

#endif
