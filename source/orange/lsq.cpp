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
#include "lsq.h"
//#include <malloc.h>

void xdisaster() {
}

void lsq::startup(int nvar, bool fit_const) {
/*
!     Allocates dimensions for arrays and initializes to zero
!     The calling program must set nvar = the number of variables, and
!     fit_const = true if a constant is to be included in the model,
!     otherwise fit_const = false
!
	!--------------------------------------------------------------------------*/
	
	int i;
	
	nobs = 0;
	if (fit_const) {
		ncol = nvar + 1;
	} else {
		ncol = nvar;
	}
	
	if (initialized) {
		free(d);
		free(rhs);
		free(r);
		free(tol);
		free(rss);
		free(vorder);
	}
	
	r_dim = ncol * (ncol - 1)/2;
	
	++ncol;
	++r_dim;
	d = (double*)malloc(sizeof(double)*ncol);
	rhs = (double*)malloc(sizeof(double)*ncol);
	tol = (double*)malloc(sizeof(double)*ncol);
	rss = (double*)malloc(sizeof(double)*ncol);
	vorder = (int*)malloc(sizeof(int)*ncol);
	r = (double*)malloc(sizeof(double)*r_dim);
	--ncol;
	--r_dim;
	
	for (i = 0; i <= ncol; ++i) {
		d[i] = zero;
		rhs[i] = zero;
	}
//nt tmpc=0;
	for (i = 0; i <= r_dim; ++i) {
/*		if (i%((ncol+ncol-tmpc)/2 + 1)==0) {
			tmpc++;
			r[i]=0.5;
		}
		else*/
			r[i] = zero;
	}
	sserr = zero;
	
	if (fit_const) {
		for (i = 1; i <= ncol; ++i) {
			vorder[i] = i-1;
		}
	} else { 	
		for (i = 1; i <= ncol; ++i) {
			vorder[i] = i;
		}
	}

	initialized = true;
	tol_set = false;
	rss_set = false;
}

void lsq::includ(double weight, double *xrow, double yelem) {
/*
!     ALGORITHM AS75.1  APPL. STATIST. (1974) VOL.23, NO. 3
!     Calling this routine updates D, R, RHS and SSERR by the
!     inclusion of xrow, yelem with the specified weight.
!     *** WARNING  Array XROW is overwritten.
!     N.B. As this routine will be called many times in most applications,
!          checks have been eliminated.
!
!--------------------------------------------------------------------------
	*/
	int i, k, nextr;
	double w, y, xi, di, wxi, dpi, cbar, sbar, xk;
	
	nobs = nobs + 1;
	w = weight;
	y = yelem;
	rss_set = false;
	nextr = 1;
	for (i = 1; i <= ncol; ++i) {
		//!     Skip unnecessary transformations.   Test on exact zeroes must be
		//!     used or stability can be destroyed.
		
		if (fabs(w) < vsmall)
			return;
		xi = xrow[i];
		if (fabs(xi) < vsmall)
			nextr = nextr + ncol - i;
		else {
			di = d[i];
			wxi = w * xi;
			dpi = di + wxi*xi;
			cbar = di / dpi;
			sbar = wxi / dpi;
			w = cbar * w;
			d[i] = dpi;
			for (k = i+1; k <= ncol; ++k) {
 				xk = xrow[k];
				xrow[k] = xk - xi * r[nextr];
				r[nextr] = cbar * r[nextr] + sbar * xk;
				nextr = nextr + 1;
			}
			xk = y;
			y = xk - xi * rhs[i];
			rhs[i] = cbar * rhs[i] + sbar * xk;
		}
	}
	
	//!     Y * sqrt(W) is now equal to the Brown, Durbin & Evans recursive
	//!     residual.
	
	sserr = sserr + w * y * y;
	
}


void lsq::regcf(double *beta, int nreq, int& ifault) {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!     Modified version of AS75.4 to calculate regression coefficients
!     for the first NREQ variables, given an orthogonal reduction from
!     AS75.1.
!
!--------------------------------------------------------------------------
	*/
	int i, j, nextr;
	
	ifault = 0;
	if (nreq < 1 || nreq > ncol) 
		ifault = ifault + 4;
	if (ifault != 0) 
		return;
	
	if (!tol_set) 
		tolset();
	
	for (i = nreq; i >= 1; --i) {
		if (sqrt(d[i]) < tol[i]) {
			beta[i] = zero;
			d[i] = zero;
		} else {
			beta[i] = rhs[i];
			nextr = (i-1) * (ncol+ncol-i)/2 + 1;
			for(j = i+1; j <= nreq; ++j) {
				beta[i] -= r[nextr] * beta[j];
				nextr = nextr + 1;
			}
		}
	}
	
	return;
}




void lsq::tolset() {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

  !     Sets up array TOL for testing for zeroes in an orthogonal
  !     reduction formed using AS75.1.
	*/
	int col, row, pos,i;
	double eps, ten = 10.0, total, *work;
	
	work = new double[ncol+1];
	eps = .2220e-7;
	
	/*
	!     Set tol(i) = sum of absolute values in column I of R after
	!     scaling each element by the square root of its row multiplier,
	!     multiplied by EPS.
	*/
	
	for (i = 1; i <= ncol; ++i) {
		work[i] = sqrt(d[i]);
	}
	for(col = 1; col <= ncol; ++col) {
		pos = col - 1;
		total = work[col];
		for(row = 1; row < col; ++row) {
			total = total + fabs(r[pos]) * work[row];
			pos = pos + ncol - row - 1;
		}
		tol[col] = eps * total;
	}
	
	tol_set = true;
	delete[] work;
}



void lsq::sing(bool* lindep, int& ifault) {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!     Checks for singularities, reports, and adjusts orthogonal
!     reductions produced by AS75.1.
!--------------------------------------------------------------------------
	*/
	
	double temp, *x, *work, y, weight;
	int col, pos, row, pos2,i,l;
	
	x = new double[ncol+1];
	work = new double[ncol+1];
	
	ifault = 0;

	if(!tol_set)
		tolset();
	
	for (i = 1; i <= ncol; ++i) {
		work[i] = sqrt(d[i]);
	}	
	for( col = 1; col <= ncol; ++col) {
		
	/*
	!     Set elements within R to zero if they are less than tol(col) in
	!     absolute value after being scaled by the square root of their row
	!     multiplier.
		*/
		
		temp = tol[col];
		pos = col - 1;
		for(row = 1; row <= col-1; ++row) {
			if (fabs(r[pos]) * work[row] < temp) 
				r[pos] = zero;
			pos = pos + ncol - row - 1;
		}
		/*
		!     If diagonal element is near zero, set it to zero, set appropriate
		!     element of LINDEP, and use INCLUD to augment the projections in
		!     the lower rows of the orthogonalization.
		*/
//		printf("%f %f\n",work[col],temp);
		lindep[col] = false;
		if (work[col] <= temp) {
			lindep[col] = true;
			ifault = ifault - 1;
			if (col < ncol) {
				pos2 = pos + ncol - col + 1;
				for(l = 1; l <= ncol; ++l) {
					x[l] = zero;
				}
				for(l = col+1; l <= ncol; ++l) {
					x[l] = r[pos+l-col];
				}
				y = rhs[col];
				weight = d[col];
				for(l = pos+1; l <= pos2-1; ++l) {
					r[l] = zero;
				}
				d[col] = zero;
				rhs[col] = zero;
				includ(weight, x, y);
				nobs--;
			}
			else
				sserr = sserr + d[col] * rhs[col]*rhs[col];
		}
	}
	
	delete[] x;
	delete[] work;
	return;
}


void lsq::ss() {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!     Calculates partial residual sums of squares from an orthogonal
!     reduction from AS75.1.
!
!--------------------------------------------------------------------------
	*/
	
	int i;
	double total;
	
	total = sserr;
	rss[ncol] = sserr;
	for (i = ncol; i >= 2; --i) {
		total = total + d[i] * rhs[i]*rhs[i];
		rss[i-1] = total;
	}
	rss_set = false;
	return;
}

void lsq::cov(int nreq, double& var, double *covmat, int dimcov, double *sterr, int& ifault) {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!     Calculate covariance matrix for regression coefficients for the
!     first nreq variables, from an orthogonal reduction produced from
!     AS75.1.
	*/
	
	int dim_rinv, pos, row, start, pos2, col, pos1, k;
	double total;
	double *rinv;
	
	//!     Check that dimension of array covmat is adequate.
	
	if (dimcov < nreq*(nreq+1)/2) {
		ifault = 1;
		return;
	}

	if (!rss_set)
		ss();

	
	//!     Check for small or zero multipliers on the diagonal.
	
	ifault = 0;
	for (row = 1;row<= nreq; ++row) {
		if (fabs(d[row]) < vsmall) 
			ifault = -row;
	}
	if (ifault != 0) 
		return;
	
	//!     Calculate estimate of the residual variance.
	
	if (nobs > nreq) 
		var = rss[nreq] / (nobs - nreq);
	else {
		ifault = 2;
		return;
	}
	
	dim_rinv = nreq*(nreq-1)/2;

	rinv = (double *)malloc(sizeof(double)*(dim_rinv+1));	

	inv(nreq, rinv);
	pos = 1;
	start = 1;
	for(row = 1; row <= nreq; ++row) {
		pos2 = start;
		for(col = row; col <= nreq; ++col) {
			pos1 = start + col - row;
			if (row == col) 
				total = one / d[col];
			else
				total = rinv[pos1-1] / d[col];
			
			for(k = col+1; k <= nreq; ++k) {
				total = total + rinv[pos1] * rinv[pos2] / d[k];
				pos1 = pos1 + 1;
				pos2 = pos2 + 1;
			}
			covmat[pos] = total * var;
			if (row == col)
				sterr[row] = sqrt(covmat[pos]);
			pos = pos + 1;
		}
		start = start + nreq - row;
	}
	free(rinv);
	return;
}


void lsq::inv(int nreq, double *rinv) {
	
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2
!     Invert first nreq rows and columns of Cholesky factorization
!     produced by AS 75.1.
!
!--------------------------------------------------------------------------
	*/
	
	int pos, row, col, start, k, pos1, pos2;
	double total;
	
	//!     Invert R ignoring row multipliers, from the bottom up.
	
	pos = nreq * (nreq-1)/2;
	for (row = nreq-1; row >=  1; row--) {
		start = (row-1) * (ncol+ncol-row)/2 + 1;
		for (col = nreq; col >= row+1; col--) {
			pos1 = start;
			pos2 = pos;
			total = zero;
			for( k = row+1; k <= col-1; ++k) {
				pos2 = pos2 + nreq - k;
				total = total - r[pos1] * rinv[pos2];
				pos1 = pos1 + 1;
			}
			rinv[pos] = total - r[pos1];
			pos = pos - 1;
		}
	}
}


void lsq::partial_corr(int in, double *cormat, int dimc, double *ycorr, int& ifault) {
/*
!     Replaces subroutines PCORR and COR of:
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

  !     Calculate partial correlations after the variables in rows
  !     1, 2, ..., IN have been forced into the regression.
  !     If IN = 1, and the first row of R represents a constant in the
  !     model, then the usual simple correlations are returned.
  !	    If IN = 0, the value returned in array CORMAT for the correlation
  !     of variables Xi & Xj is:
  !       sum ( Xi.Xj ) / sqrt ( sum (Xi^2) . sum (Xj^2) )
  !     On return, array CORMAT contains the upper triangle of the matrix of
  !     partial correlations stored by rows, excluding the 1's on the diagonal.
  !     e.g. if IN = 2, the consecutive elements returned are:
  !     (3,4) (3,5) ... (3,ncol), (4,5) (4,6) ... (4,ncol), etc.
  !     Array YCORR stores the partial correlations with the Y-variable
  !     starting with YCORR(IN+1) = partial correlation with the variable in
  !     position (IN+1).
  !
  !--------------------------------------------------------------------------
	*/
	
	
	int base_pos, pos, row, col, col1, col2, pos1, pos2,k;
	double *rms, sumxx, sumxy, sumyy, *work;
	
	rms = new double[ncol+1];
	work = new double[ncol+1];
	
	
	ifault = 0;
	if (in < 0 || in > ncol-1) ifault = ifault + 4;
	if (dimc < (ncol-in)*(ncol-in-1)/2) ifault = ifault + 8;
	if (ifault != 0) return;
	
	//!     Base position for calculating positions of elements in row (IN+1) of R.
	
	base_pos = in*ncol - (in+1)*(in+2)/2;
	
	
	//!     Calculate 1/RMS of elements in columns from IN to (ncol-1).
	
	if (d[in+1] > zero) 
		rms[in+1] = one / sqrt(d[in+1]);
	for (col = in+2;col<= ncol; ++col) {
		pos = base_pos + col;
		sumxx = d[col];
		for(row = in+1; row <= col-1; ++row) {
			sumxx = sumxx + d[row] * r[pos]*r[pos];
			pos = pos + ncol - row - 1;
		}
		if (sumxx > zero) 
			rms[col] = one / sqrt(sumxx);
		else {
			rms[col] = zero;
			ifault = -col;
			
		}
	}
	
	//    Calculate 1/RMS for the Y-variable
	
	sumyy = sserr;
	for(row = in+1;row <= ncol;++row) {
		sumyy = sumyy + d[row]* rhs[row]*rhs[row];
	}
	if (sumyy > zero) sumyy = one / sqrt(sumyy);
	
	/*
	!     Calculate sums of cross-products.
	!     These are obtained by taking dot products of pairs of columns of R,
	!     but with the product for each row multiplied by the row multiplier
	!     in array D.
	*/
	pos = 1;
	for (col1 = in+1; col1 <= ncol; ++col1) {
		sumxy = zero;
		for (k = col1+1; k <= ncol; ++k) {
			work[k] = zero;
		}
		pos1 = base_pos + col1;
		for(row = in+1; row <= col1-1; ++row) {
			pos2 = pos1 + 1;
			for(col2 = col1+1; col2 <= ncol; ++col2) {
				work[col2] = work[col2] + d[row] * r[pos1] * r[pos2];	  
				pos2 = pos2 + 1;
			}
			sumxy = sumxy + d[row] * r[pos1] * rhs[row];
			pos1 = pos1 + ncol - row - 1;
		}
		
		//!     Row COL1 has an implicit 1 as its first element (in column COL1)
		
		pos2 = pos1 + 1;
		for (col2 = col1+1; col2<= ncol; ++col2) {
			work[col2] = work[col2] + d[col1] * r[pos2];
			pos2 = pos2 + 1;
			cormat[pos] = work[col2] * rms[col1] * rms[col2];
			pos = pos + 1;
		}
		sumxy = sumxy + d[col1] * rhs[col1];
		ycorr[col1] = sumxy * rms[col1] * sumyy;
	}
	
	for (k = 1; k <= in; ++k) {
		ycorr[k] = zero;
	}
	
	return;
}




void lsq::vmove(int from, int to, int& ifault) {
/*
!     ALGORITHM AS274 APPL. STATIST. (1992) VOL.41, NO. 2
!     Move variable from position FROM to position TO in an
!     orthogonal reduction produced by AS75.1.
!
!--------------------------------------------------------------------------
	*/
	
	double d1, d2, x, d1new, d2new, cbar, sbar, y;
	int m, first, last, inc, m1, m2, mp1, col, pos, row,k;
	
	//!     Check input parameters
	
	ifault = 0;
	if (from < 1 || from > ncol) ifault = ifault + 4;
	if (to < 1 || to > ncol) ifault = ifault + 8;
	if (ifault != 0) return;
	
	if (from == to) {
		return;
	}
	
	if (!rss_set) ss();
	
	if (from < to) {
		first = from;
		last = to;
		inc = 1;
	} else {
		first = from - 1;
		last = to-1;
		inc = -1;
	}
	
	for (m = first; m != last; m += inc) { 
		
		//!     Find addresses of first elements of R in rows M and (M+1).
		
		m1 = (m-1)*(ncol+ncol-m)/2 + 1;
		m2 = m1 + ncol - m;
		mp1 = m + 1;
		d1 = d[m];
		d2 = d[mp1];
		
		//!     Special cases.
		
		if ((d1 < vsmall && d2 < vsmall)) {
			goto l40;
		}
		x = r[m1];
		if (fabs(x) * sqrt(d1) < tol[mp1]) {
			x = zero;
		}
		if (d1 < vsmall || fabs(x) < vsmall) {
			d[m] = d2;
			d[mp1] = d1;
			r[m1] = zero;
			for(col = m+2;col<= ncol; ++col) {
				m1 = m1 + 1;
				x = r[m1];
				r[m1] = r[m2];
				r[m2] = x;
				m2 = m2 + 1;
			}
			x = rhs[m];
			rhs[m] = rhs[mp1];
			rhs[mp1] = x;
		} else if (d2 < vsmall) {
			d[m] = d1 * x*x;
			r[m1] = one / x;
			for (k = m1+1; k <=m1+ncol-m-1; ++k) {
				r[k] /= x;
			}
			rhs[m] = rhs[m] / x;
		} else {
			l40:
			
			//!
			//!     Planar rotation in regular case.
			//!
			d1new = d2 + d1*x*x;
			cbar = d2 / d1new;
			sbar = x * d1 / d1new;
			d2new = d1 * cbar;
			d[m] = d1new;
			d[mp1] = d2new;
			r[m1] = sbar;
			for(col = m+2; col <= ncol; ++col) {
				m1 = m1 + 1;
				y = r[m1];
				r[m1] = cbar*r[m2] + sbar*y;
				r[m2] = y - x*r[m2];
				m2 = m2 + 1;
			}
			y = rhs[m];
			rhs[m] = cbar*rhs[mp1] + sbar*y;
			rhs[mp1] = y - x*rhs[mp1];
		}
		//!     Swap columns M and (M+1) down to row (M-1).
		
		if (m != 1) {
			pos = m;
			for (row = 1; row<= m-1; ++row) {
				x = r[pos];
				r[pos] = r[pos-1];
				r[pos-1] = x;
				pos = pos + ncol - row - 1;
			}
			
			//!     Adjust variable order (VORDER), the tolerances (TOL) and
			//!     the vector of residual sums of squares (RSS).
			
			m1 = vorder[m];
			vorder[m] = vorder[mp1];
			vorder[mp1] = m1;
			//printf("%i %i %f %f\n",m,mp1,tol[m],tol[mp1]);
			x = tol[m];
			tol[m] = tol[mp1];
			tol[mp1] = x;
			rss[m] = rss[mp1] + d[mp1] * rhs[mp1]*rhs[mp1];
		}
	}
	
	return;
}


void lsq::reordr(int *list, int n, int pos1, int& ifault) {
/*
!     ALGORITHM AS274  APPL. STATIST. (1992) VOL.41, NO. 2

  !     Re-order the variables in an orthogonal reduction produced by
  !     AS75.1 so that the N variables in LIST start at position POS1,
  !     though will not necessarily be in the same order as in LIST.
	!     Any variables in VORDER before position POS1 are not moved.		*/
	int next, i, l, j;
	
	//!     Check N.
	
	ifault = 0;
	if (n < 1 || n > ncol+1-pos1) ifault = ifault + 4;
	if (ifault != 0) return;
	
	//!     Work through VORDER finding variables which are in LIST.
	
	next = pos1;
	i = pos1;
	l10:   l = vorder[i];
	for(j = 1; j <= n; ++j) {
		if (l == list[j]) goto l40;
	}
	l30:
	i = i + 1;
	if (i <= ncol) goto l10;
	
	//!     If this point is reached, one or more variables in LIST has not
	//!     been found.
	
	ifault = 8;
	return;
	
	//!     Variable L is in LIST; move it up to position NEXT if it is not
	//!     already there.
	
	l40:
	if (i > next) 
		vmove(i, next,ifault);
	next = next + 1;
	if (next < n+pos1) goto l30;
	
	return;
}



void lsq::hdiag(double *xrow, int nreq, double& hii,int& ifault) {
	int  col, row, pos;
	double total, *wk;
	
	//!     Some checks
	
	ifault = 0;
	if (nreq > ncol) ifault = ifault + 4;
	if (ifault != 0) {
		return;
	}
	
	wk = new double[ncol+1];
	
	//!     The elements of xrow.inv(R).sqrt(D) are calculated and stored
	//!     in WK.
	
	hii = zero;
	for( col = 1; col <= nreq; ++col) {
		if (sqrt(d[col]) <= tol[col]) {
			wk[col] = zero;
		} 
		else {
			pos = col - 1;
			total = xrow[col];
			for (row = 1; row <= col-1; ++row) {
				total = total - wk[row]*r[pos];
				pos = pos + ncol - row - 1;
			}
			wk[col] = total;
			hii = hii + total*total / d[col];
		}
	}
	delete[] wk;
	return;
}		


double lsq::varprd(double *x, int nreq, double& var, int& ifault) {
	
	//!     Calculate the variance of x'b where b consists of the first nreq
	//!     least-squares regression coefficients.
	
	int  row;
	double *wk, rvarprd;
	
	//!     Check input parameter values
	
	rvarprd = zero;
	ifault = 0;
	if (nreq < 1 || nreq > ncol) ifault = ifault + 4;
	if (nobs <= nreq) ifault = ifault + 8;
	if (ifault != 0) 
		return 0.0;
	
	wk = new double[nreq+1];
	
	//!     Calculate the residual variance estimate.
	
	var = sserr / (nobs - nreq);
	
	//!     Variance of x'b = var.x'(inv R)(inv D)(inv R')x
	//!     First call BKSUB2 to calculate (inv R')x by back-substitution.
	
	bksub2(x, wk, nreq);
	for( row = 1; row <= nreq; ++row) {
		if(d[row] > tol[row]) 
			rvarprd += wk[row]*wk[row] / d[row];
	}
	rvarprd = rvarprd * var;
	
	delete[] wk;
	
	return rvarprd;
}



void lsq::bksub2(double *x, double *b, int nreq) {
	
	//!     Solve x = R'b for b given x, using only the first nreq rows and
	//!     columns of R, and only the first nreq elements of R.
	
	int  pos, row, col;
	double temp;
	
	//!     Solve by back-substitution, starting from the top.
	
	for( row = 1; row<= nreq; ++row) {
		pos = row - 1;
		temp = x[row];
		for(col = 1; col <= row-1; ++col) {
			temp = temp - r[pos]*b[col];
			pos = pos + ncol - col - 1;
		}
		b[row] = temp;
	}
}
