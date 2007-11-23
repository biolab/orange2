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
    Contact: miha.stajdohar@fri.uni-lj.si
*/


#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "Python.h"

#include <stdio.h>
#include "stdlib.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <queue>
#include <vector>
#include <math.h>
#include <string>
#include <time.h>

#include "px/orangeom_globals.hpp"
#include "root.hpp"
#include "graph.hpp"
#include "table.hpp"
#include "stringvars.hpp"

#ifdef DARWIN
#include <strings.h>
#endif

using namespace std;

WRAPPER(ExampleTable)
OMWRAPPER(Network)

class ORANGEOM_API TNetwork : public TGraphAsList
{
public:
  __REGISTER_CLASS
  
  TNetwork(TGraphAsList *graph);
  TNetwork(const int &nVert, const int &nEdge, const bool dir);
  ~TNetwork();

  PExampleTable items; //P ExampleTable of vertices data
}; 

#endif
