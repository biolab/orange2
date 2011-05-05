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


#ifdef _MSC_VER
  #pragma warning (disable : 4290)
#endif

#include "orvector.ppp"

DEFINE_TOrangeVector_classDescription(bool, "TBoolList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(char, "TBoolList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(int, "TIntList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(long, "TLongList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(float, "TFloatList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(double, "TDoubleList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(string, "TStringList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PFloatList, "TFloatListList", true, ORANGE_API)


DEFINE_AttributedList_classDescription(TAttributedFloatList, TFloatList)
DEFINE_AttributedList_classDescription(TAttributedBoolList, TBoolList)

// TValueList's properties are defined in vars.cpp

#define pff pair<float, float>
DEFINE_TOrangeVector_classDescription(pff, "TFloatFloatList", false, ORANGE_API)

#define pif pair<int, float>
DEFINE_TOrangeVector_classDescription(pif, "TIntFloatList", false, ORANGE_API)

#ifdef _MSC_VER_60
TClassDescription template TOrangeVector<TValue, false>::st_classDescription;
ORANGE_EXTERN template class ORANGE_API TOrangeVector<TValue, false>;
#else
 DEFINE_TOrangeVector_classDescription(TValue, "TOrangeVector<TValue, false>", false, ORANGE_API)
//template<> TClassDescription TOrangeVector<TValue, false>::st_classDescription; // =  = { "StringList", &typeid(TValueList), &TOrange::st_classDescription, TOrange_properties, TOrange_components };
#endif



/* This function is stolen from Python 2.3 (file listobject.c):
   n between 2^m-1 and 2^m, is round up to a multiple of 2^(m-5) */
int _RoundUpSize(const int &n)
{
	unsigned int nbits = 0;
	unsigned int n2 = (unsigned int)n >> 5;
	do {
		n2 >>= 3;
		nbits += 3;
	} while (n2);
	return ((n >> nbits) + 1) << nbits;
}
