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
