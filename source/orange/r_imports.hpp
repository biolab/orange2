#ifndef __R_IMPORTS_HPP
#define __R_IMPORTS_HPP

#ifndef NO_JIT_LINKING

extern "C" {
  extern void (*i__dqrls)(double *x, int *n, int *p, double *y, int *ny, double *tol, double *b, double *rsd, double *qty, int *k, int *jpvt, double *qraux, double *work); //AS dqrls_
  extern void (*i__chol2inv)(double *x, int *size); //AS La_chol2inv
};

#include "r_imports.ipp"

#else

  extern "C" {
  __declspec(dllimport) void dqrls_(double *x, int *n, int *p, double *y, int *ny, double *tol, double *b, double *rsd, double *qty, int *k, int *jpvt, double *qraux, double *work); //AS dqrls_
  __declspec(dllimport) void chol2inv(double *x, int *size); //AS La_chol2inv
  };

  #define dqrls dqrls_

  #pragma lib("C:\Program Files\R\rw2001\lib\Rdll.lib")
#endif

#endif
