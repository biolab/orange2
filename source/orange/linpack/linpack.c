/*
 * Minimal environment for output from f2c to compile and run.
 */

#include "linpack.h"

double d_sign(double * A, double * B)
{
	double x = (*A >= 0)? *A : -*A;
	return (*B >= 0)? x : -x;
}
