/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef __RANDOM_HPP
#define __RANDOM_HPP

#include <string>
#include "root.hpp"
using namespace std;

#include "cMersenneTwister.h"

class ORANGE_API cMersenneTwister;

WRAPPER(RandomGenerator)

class ORANGE_API TRandomGenerator : public TOrange {
public:
    __REGISTER_CLASS

    int initseed; //P initial random seed
    int uses;     //P #times a number was returned

    cMersenneTwister mt;

    TRandomGenerator(const int &aninitseed=0)
      : initseed(aninitseed),
        uses(0),
        mt((unsigned long)aninitseed << 1)  // multiply by 2 since the generator sets the 0th bit
      {}

    virtual void reset()      
      { uses=0; 
        mt.Init((unsigned long)initseed << 1); }   // multiply by 2 since the generator sets the 0th bit

    inline unsigned long operator()() 
      { uses++;
        return mt.Random(); }


    #define crand ( (unsigned long)operator()() )
    #define irand ( (unsigned int)operator()() )

    int randbool(const unsigned int &y=2)
    { return (irand % y) == 0; }

    inline int randint()
    { return int(irand >> 1); }

    inline int randint(const unsigned int &y)
    { return int(irand % y); }

    inline int randint(const int &x, const int &y)
    { return int(irand % ((unsigned int)(y-x+1)) + x); }

    inline long randlong()
    { return long(crand >> 1); }

    inline long randlong(const unsigned long y)
    { return crand % y; }

    inline long randlong(const long &x, const long &y)
    { return crand % ((unsigned long)(y-x+1)) + x; }

    inline double randdouble(const double &x, const double &y)
    { return double(crand)/double(4294967296.0)*(y-x) + x; }

    inline float randfloat(const double &x, const double &y)
    { return float(randdouble(x, y)); }

    inline float operator()(const double &x, const double &y)
    { return randdouble(x, y); }

    inline double randdouble(const double &y=1.0)
    { return double(crand)/double(4294967296.0)*y; }

    inline float randfloat(const float &y=1.0)
    { return float(randdouble(y)); }
};


// Same as or_random_shuffle, but uses TRandomGenerator
template<typename RandomAccessIter>
void rg_random_shuffle(RandomAccessIter first, RandomAccessIter last, TRandomGenerator &rand)
{
  if (first == last)
    return;
  
  for (RandomAccessIter i = first + 1; i != last; ++i)
    iter_swap(i, first + rand.randint((i - first)));
}
    

/* globalRandom is wrapped _globalRandom. Use any of them, they are same... */
extern TRandomGenerator *_globalRandom;
extern PRandomGenerator globalRandom;


/* This is to be used when you only need a few random numbers and don't want to
   initialize the Mersenne twister.

   DO NOT USE THIS WHEN YOU NEED 32-BIT RANDOM NUMBERS!

   The below formula is the same as used in MS VC 6.0 library. */

class ORANGE_API TSimpleRandomGenerator {
public:
  unsigned int seed;
  
  TSimpleRandomGenerator(int aseed = 0)
  : seed(aseed)
  {}

  unsigned int rand ()
  { return (((seed = seed * 214013L + 2531011L) >> 16) & 0x7fff); }

  int operator()(const int &y)
  { return rand() % y; }
  
  int randbool(const int &y=2)
  { return (rand()%y) == 0; }

  int randsemilong()
  { return rand()<<15 | rand(); }

  float randfloat()
  { return float(rand())/0x7fff; }
};

#endif
