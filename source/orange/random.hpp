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
      : initseed(aninitseed), uses(0),
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


class TStdRandomGenerator : public TRandomGenerator
{ public:
    TStdRandomGenerator()
      { srand(0); }
    void reset()
      {}
};

/* globalRandom is wrapped _globalRandom. Use any of them, they are same... */
extern TStdRandomGenerator *_globalRandom;
extern PRandomGenerator globalRandom;

#define msrand *_globalRandom.mt.Init
#define mrand abs(*_globalRandom())

// returns true with probability 1/y
inline int randbool(int y=2)       
{ return _globalRandom->randbool(y); }

inline int randint()
{ return _globalRandom->randint(); }

inline int randint(int y)
{ return _globalRandom->randint(y); }

inline int randint(int x, int y)
{ return _globalRandom->randint(x, y); }

inline long randlong()
{ return _globalRandom->randlong(); }

inline long randlong(long y)
{ return _globalRandom->randlong(y); }

inline long randlong(long x, long y)
{ return _globalRandom->randlong(x, y); }

inline double randdouble(double x, double y)
{ return _globalRandom->randdouble(x, y); }

inline float randfloat(double x, double y)
{ return _globalRandom->randfloat(x, y); }

inline double randdouble(double y=1.0)
{ return _globalRandom->randdouble(y); }

inline float randfloat(double y=1.0)
{ return _globalRandom->randfloat(y); }

#endif
