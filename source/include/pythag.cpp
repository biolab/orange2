#include <math.h>

namespace nrutil {

    #define NRANSI

    #include "nrutil.h"

    #pragma warning(disable: 4786 4244) // warning C4786: symbol greater than 255 characters

    float pythag(float a, float b)
    {
	    float absa,absb;
	    absa=fabs(a);
	    absb=fabs(b);
	    if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
	    else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
    }

};

#undef NRANSI