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


#ifdef _MSC_VER
  #pragma warning (disable : 4290)
#endif

#include "orvector.hpp"

DEFINE__TOrangeVector_classDescription(bool, "TBoolList")
DEFINE__TOrangeVector_classDescription(char, "TBoolList")
DEFINE__TOrangeVector_classDescription(int, "TIntList")
DEFINE__TOrangeVector_classDescription(long, "TLongList")
DEFINE__TOrangeVector_classDescription(float, "TFloatList")
DEFINE__TOrangeVector_classDescription(double, "TDoubleList")
DEFINE__TOrangeVector_classDescription(string, "TStringList")

// TValueList's properties are defined in vars.cpp
// TAttributeFloatList's properties are defined in vars.cpp

#define pff pair<float, float>
DEFINE__TOrangeVector_classDescription(pff, "TFloatFloatList")

#define pif pair<int, float>
DEFINE__TOrangeVector_classDescription(pif, "TIntFloatList")
