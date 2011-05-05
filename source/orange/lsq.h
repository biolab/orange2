/* 
The module LSQ is for unconstrained linear least-squares fitting.   It is
based upon Applied Statistics algorithm AS 274 (see comments at the start
of the module).    A planar-rotation algorithm is used to update the QR-
factorization.   This makes it suitable for updating regressions as more
data become available.   The module contains a test for singularities which
is simpler and quicker than calculating the singular-value decomposition.


This package is being made available freely, and may be freely distributed.
The author is:
     Author: Alan Miller
             CSIRO Division of Mathematics & Statistics
             Private Bag 10, Rosebank MDC
             Clayton 3169, Victoria, Australia
     Phone: (+61) 3 9545-8036      Fax: (+61) 3 9545-8080
     e-mail: Alan.Miller @ mel.dms.csiro.au
     
*/

#include <math.h>
#include <stdlib.h>

const double zero = 0.0;
const double one = 1.0;
const double vsmall = 1.e-69;


class lsq {
	
/*
!     The PUBLIC variables are:
!     lsq_kind = a KIND parameter for the floating-point quantities calculated
!                in this module.   See the more detailed explanation below.
!                This KIND parameter should be used for all floating-point
!                arguments passed to routines in this module.
!     nobs    = the number of observations processed to date.
!     ncol    = the total number of variables, including one for the constant,
!               if a constant is being fitted.
!     r_dim   = the dimension of array r = ncol*(ncol-1)/2
!     vorder  = an integer vector storing the current order of the variables
!               in the QR-factorization.   The initial order is 0, 1, 2, ...
!               if a constant is being fitted, or 1, 2, ... otherwise.
!     initialized = a logical variable which indicates whether space has
!                   been allocated for various arrays.
!     tol_set = a logical variable which is set when subroutine TOLSET has
!               been called to calculate tolerances for use in testing for
!               singularities.
!     rss_set = a logical variable indicating whether residual sums of squares
!               are available and usable.
!     d()     = array of row multipliers for the Cholesky factorization.
!               The factorization is X = Q.sqrt(D).R where Q is an ortho-
!               normal matrix which is NOT stored, D is a diagonal matrix
!               whose diagonal elements are stored in array d, and R is an
!               upper-triangular matrix with 1's as its diagonal elements.
!     rhs()   = vector of RHS projections (after scaling by sqrt(D)).
!               Thus Q'y = sqrt(D).rhs
!     r()     = the upper-triangular matrix R.   The upper triangle only,
!               excluding the implicit 1's on the diagonal, are stored by
!               rows.
!     tol()   = array of tolerances used in testing for singularities.
!     rss()   = array of residual sums of squares.   rss(i) is the residual
!               sum of squares with the first r variables in the model.
!               By changing the order of variables, the residual sums of
!               squares can be found for all possible subsets of the variables.
!               The residual sum of squares with NO variables in the model,
!               that is the total sum of squares of the y-values, can be
!               calculated as rss(1) + d(1)*rhs(1)^2.   If the first variable
!               is a constant, then rss(1) is the sum of squares of
!               (y - ybar) where ybar is the average value of y.
!     sserr   = residual sum of squares with all of the variables included.
	*/
	
public:
	
	int nobs, ncol, r_dim;
	int *vorder;
	bool initialized ;
	bool tol_set ;
	bool rss_set ;
	
	/*
	!     Note. lsq_kind is being set to give at least 10 decimal digit
	!           representation of floating point numbers.   This should be
	!           adequate for most problems except the fitting of polynomials.
	!           lsq_kind is being set so that the same code can be run on PCs
	!           and Unix systems, which will usually represent floating-point
	!           numbers in `double precision', and other systems with larger
	!           word lengths which will give similar accuracy in `single
	!           precision'.
	*/
	
	double *d, *rhs, *r, *tol, *rss;
	double sserr;

	lsq() {
		initialized = false;
		tol_set = false;
		rss_set = false;
	}
	
	~lsq() {
		if (initialized) {
			free(d);
			free(rhs);
			free(r);
			free(tol);
			free(rss);
			free(vorder);
		}
	}
	 
	void startup(int nvar, bool fit_const);
	void includ(double weight, double *xrow, double yelem);
	void regcf(double *beta, int nreq, int& ifault);
	void tolset();
	void sing(bool *lindep, int& ifault);
	void ss();
	void cov(int nreq, double& var, double *covmat, int dimcov, double *sterr, int& ifault);
	void inv(int nreq, double *rinv);
	void partial_corr(int in, double *cormat, int dimc, double *ycorr,int& ifault);
	void vmove(int from, int to,int& ifault);
	void reordr(int *list, int n, int pos1, int& ifault);
	void hdiag(double *xrow, int nreq, double& hii, int& ifault);	
	double varprd(double *x, int nreq, double& var, int& ifault);
	void bksub2(double *x, double *b, int nreq);
};

