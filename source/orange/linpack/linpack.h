/*
 * Minimal environment for output from f2c to compile and run.
 */

#ifndef LINPACK_H
#define LINPACK_H

#define doublereal double
#define integer int
#define logical char

#ifndef MAX
#define MAX(a,b)    (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#endif

#define max MAX
#define min MIN

#ifdef __cplusplus
extern "C" {
#endif

	double d_sign(doublereal * A, doublereal * B);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
//	#include "../blas/blas.h"
	extern double dnrm2_(int *, double *, int *);
	extern double ddot_(int *, double *, int *, double *, int *);
	extern int daxpy_(int *, double *, double *, int *, double *, int *);
	extern int dscal_(int *, double *, double *, int *);
#ifdef __cplusplus
}
#endif

#endif //LINPACK_H
