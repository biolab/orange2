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


#include "boolcnt.hpp"
#include "stladdon.hpp"
#include <algorithm>

using namespace std;

/* TCounter member functions */

TCounter::TCounter(int nums, int alimit)
 : vector<int>(nums), limit(alimit)
{ reset(); 
}

bool TCounter::reset() 
{ generate(begin(), end(), TGenInt<int>());
  return (int(size())<=limit);
}

bool TCounter::next()
{ int limta=limit;
  iterator li=end();
  while(( (++(*(--li))) == (limta--)) && (li!=begin()));
  if (*li==limta+1) { 
    generate(begin(), end(), TGenInt<int>(limit-size())); 
    return 0;
  }
  for(;++li != end(); *li= *(li-1)+1);
  return 1;
}

bool TCounter::prev()
{ iterator li=end();
  int prev = -2; // initialized to avoid warning
  while(li!=begin()) {
    prev = (--li == begin() ) ? -1 : *(li-1);
    if (-- *li != prev) break;
  }
  if (*li==prev) { 
    reset(); 
    return 0;
  }
  generate(li+1, end(), TGenInt<int>(limit- (end()-li)+1)); 
  return 1;
}


/* TBoolCounters member functions */

TBoolCounters::TBoolCounters(int bits) 
  : vector<unsigned char>(bits, 0)
{}

TBoolCounters::~TBoolCounters() 
{}

int TBoolCounters::bitsOn()
{ return count(begin(), end(), 1);
}



/* TBoolCount_const member functions */

TBoolCount_const::TBoolCount_const(const vector<unsigned char> &ab)
  : TBoolCounters(ab.size())
{ vector<unsigned char>::const_iterator ai=ab.begin();
  for(iterator bi=begin(); bi!=end(); *(bi++)=*(ai++));
}

TBoolCount_const::TBoolCount_const(const int n) : TBoolCounters(n) {}


bool TBoolCount_const::next() 
{ return 0; }

bool TBoolCount_const::prev()
{ return 0; }


/* TBoolCount_n member functions */

TBoolCount_n::TBoolCount_n(int bits, int aset) 
 : TBoolCounters(bits), counter(aset, bits)
{ synchro(); }


void TBoolCount_n::synchro()
{ fill(begin(), end(), 0); 
  ITERATE(TCounter, ci, counter) *(begin()+ *ci)=1;
}

bool TBoolCount_n::next() 
{ bool wp=counter.next(); 
  synchro(); 
  return wp;
}

bool TBoolCount_n::prev() 
{ bool wp=counter.prev(); 
  synchro(); 
  return wp;
}

int TBoolCount_n::bitsOn() 
{ return counter.size(); 
}



/* TBoolCount member functions */

TBoolCount::TBoolCount(int abits)
 : TBoolCounters(abits)
{}


unsigned char TBoolCount::neg(unsigned char &b)
{ b= !b; return b;
}

bool TBoolCount::next()
{ if (!size())
    return 0;
  
  iterator bi = end(); 
  while (!neg(*(--bi)) && (bi!=begin()));
  if (*bi)
    return 1;

  fill(begin(), end(), 1);
  return 0;
}

bool TBoolCount::prev()
{ if (!size())
    return 0;

  iterator bi = end(); 
  while(neg(*(--bi)) && (bi!=begin()));
  if (!*bi)
    return 1;

  fill(begin(), end(), 0);
  return 0;
}


TLimitsCounter::TLimitsCounter(const vector<int> &alimits)
 : limits(alimits)
 { reset(); }


bool TLimitsCounter::reset()
{ clear();
  for(int i = limits.size(); i--; )
    push_back(0);
  return true;
}

bool TLimitsCounter::next()
{ int i = size();
  while(i-- && ((at(i) = (++at(i)) % limits[i]) == 0));
  if (i==-1) { reset(); return false; }
  return true;
}

bool TLimitsCounter::prev()
{ int i = size(); 
  while(i-- && (at(i)-- || (at(i)=limits[i]-1 > 0)));
  if (i==-1) { reset(); return false; }
  return true;
}

