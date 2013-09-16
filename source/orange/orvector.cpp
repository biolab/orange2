#ifdef _MSC_VER
  #pragma warning (disable : 4290)
#endif

#include "orvector.hpp"

DEFINE_TOrangeVector_classDescription(bool, "TBoolList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(char, "TBoolList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(int, "TIntList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(long, "TLongList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(float, "TFloatList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(double, "TDoubleList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(string, "TStringList", false, ORANGE_API)
DEFINE_TOrangeVector_classDescription(PFloatList, "TFloatListList", true, ORANGE_API)


#define pff pair<float, float>
DEFINE_TOrangeVector_classDescription(pff, "TFloatFloatList", false, ORANGE_API)

#define pif pair<int, float>
DEFINE_TOrangeVector_classDescription(pif, "TIntFloatList", false, ORANGE_API)


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
