/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
*/


#ifndef __RANDOM_HPP
#define __RANDOM_HPP

#include <string>
#include "garbage.hpp"
#include "root.hpp"
using namespace std;

#include "cMersenneTwister.h"

WRAPPER(RandomGenerator)

class TRandomGenerator : public TOrange {
public:
    __REGISTER_CLASS

    int initseed; //P initial random seed
    int uses;     //P #times a number was returned

    cMersenneTwister mt;

    TRandomGenerator(const int &aninitseed=0)
      : initseed(aninitseed),
        uses(0),
        mt(long(aninitseed))
      {}

    virtual void reset()      
      { uses=0; 
        mt.Init(initseed); }

    inline unsigned long operator()() 
      { uses++;
        return mt.Random(); }


    #define crand ( (unsigned long)operator()() >> 1 )
    #define irand ( (unsigned int)operator()() >> 1 )

    int randbool(int y=2)
    { return (irand%y) == 0; }

    inline int randint()
    { return int(irand); }

    inline int randint(int y)
    { return int(irand%y); }

    inline int randint(int x, int y)
    { return int(irand%(y-x+1) + x); }

    inline long randlong()
    { return crand; }

    inline long randlong(long y)
    { return crand%y; }

    inline long randlong(long x, long y)
    { return crand%(y-x+1) + x; }

    inline double randdouble(double x, double y)
    { return double(crand)/double(2147483648.0)*(y-x) + x; }

    inline float randfloat(double x, double y)
    { return float(double(crand)/double(2147483648.0)*(y-x) + x); }

    inline float operator()(const double &x, const double &y)
    { return randfloat(x, y); }

    inline double randdouble(double y=1.0)
    { return double(crand)/double(2147483648.0)*y; }

    inline float randfloat(double y=1.0)
    { return float(double(crand)/double(2147483648.0)*y); }
};


/* globalRandom is wrapped _globalRandom. Use any of them, they are same... */
extern TRandomGenerator *_globalRandom;
extern PRandomGenerator globalRandom;

#define LOCAL_OR_GLOBAL_RANDOM (randomGenerator ? const_cast<TRandomGenerator &>(randomGenerator.getReference()) : *_globalRandom)


/* This is to be used when you only need a few random numbers and don't want to
   initialize the Mersenne twister.

   DO NOT USE THIS WHEN YOU NEED 32-BIT RANDOM NUMBERS!

   The below formula is the same as used in MS VC 6.0 library. */

class TSimpleRandomGenerator {
public:
  int seed;
  
  TSimpleRandomGenerator(int aseed)
  : seed(aseed)
  {}

  int rand ()
  { return(((seed = seed * 214013L + 2531011L) >> 16) & 0x7fff); }

  int randbool(int y=2)
  { return (rand()%y) == 0; }

  int randsemilong()
  { return rand()<<15 | rand(); }
};

#endif
