/*

This file contains code from various sources.
Its origins are described at each function separately.

*/


#include "logreg.hpp"

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

const double EPSILON = .1192e-06;
const double INF = 1e32;

double lngamma(double z) {
//  Uses Lanczos-type approximation to ln(gamma) for z > 0.
//  Reference:
//       Lanczos, C. 'A precision approximation of the gamma
//               function', J. SIAM Numer. Anal., B, 1, 86-96, 1964.
//  Accuracy: About 14 significant digits except for small regions
//            in the vicinity of 1 and 2.

	double a[9] = {0.9999999999995183, 676.5203681218835,
                      -1259.139216722289, 771.3234287757674, 
                      -176.6150291498386, 12.50734324009056, 
                      -0.1385710331296526, 0.9934937113930748e-05,
					  0.1659470187408462e-06};
	double lnsqrt2pi = 0.9189385332046727;
	double lanczos,tmp;

	int j;

	if (z <= EPSILON)
		return 0;

	lanczos = 0;
	tmp = z + 7;
	for (j = 8; j >= 1; --j) {
	  lanczos = lanczos + a[j]/tmp;
	  tmp = tmp - 1;
	}
	lanczos = lanczos + a[0];
	lanczos = log(lanczos) + lnsqrt2pi - (z + 6.5) + (z - 0.5)*log(z + 6.5);

	return lanczos;
}



double alnorm(double x, bool upper) {
	double norm_prob, con=1.28, z,y, ltone = 7.0, utzero = 18.66;
	double p, r, a2, b1, c1, c3, c5, d1, d3, d5;
	double q, a1, a3, b2, c2, c4, c6, d2, d4;
	bool up;

//  Algorithm AS66 Applied Statistics (1973) vol.22, no.3

// Evaluates the tail area of the standardised normal curve
// from x to infinity if upper is .true. or
// from minus infinity to x if upper is .false.

	
	p = 0.398942280444;
	q = 0.39990348504;
	r = 0.398942280385; 
	a1 = 5.75885480458;
	a2 = 2.62433121679; 
	a3 = 5.92885724438; 
	b1 = -29.8213557807;
	b2 = 48.6959930692;
    c1 = -3.8052e-8;
	c2 = 3.98064794e-4;       
    c3 = -0.151679116635;
	c4 = 4.8385912808;
	c5 = 0.742380924027;
	c6 = 3.99019417011; 
    d1 = 1.00000615302; 
	d2 = 1.98615381364;  
	d3 = 5.29330324926;
	d4 = -15.1508972451;
    d5 = 30.789933034;

	up = upper;
	z = x;
	if(z < 0.0) {
		up = !up;
		z = -z;
	}
	if(z <= ltone || up && z <= utzero) {
		y = 0.5*z*z;
	} else {
		norm_prob = 0.0;
	}
	if(z > con) {
		norm_prob = r*exp(-y)/(z+c1+d1/(z+c2+d2/(z+c3+d3/(z+c4+d4/(z+c5+d5/(z+c6))))));
	} else {
		norm_prob = 0.5 - z*(p-q*y/(y+a1+b1/(y+a2+b2/(y+a3))));
	}
	if(!up) 
		norm_prob = 1.0 - norm_prob;

	return norm_prob;
}


double gammad(double x, double p) {
	//!  ALGORITHM AS239  APPL. STATIST. (1988) VOL. 37, NO. 3
	//!  Computation of the Incomplete Gamma Integral
	//!  Auxiliary functions required: ALNORM = algorithm AS66 (included) & LNGAMMA
	//!  Converted to be compatible with ELF90 by Alan Miller
	//!  N.B. The return parameter IFAULT has been removed as ELF90 allows only
	//!  one output parameter from functions.   An error message is issued instead.
	
	double gamma_prob;
	double pn1, pn2, pn3, pn4, pn5, pn6, tol = 1.e-14, oflo = 1.e+37;
	double xbig = 1.e+8, arg, c, rn, a, b, one = 1.0, zero = 0.0, an;
	double two = 2.0, elimit = -88.0, plimit = 1000.0, three = 3.0;
    double nine = 9;
	
	gamma_prob = zero;
	
	if	(p <= zero || x < EPSILON) {
		return 0.0;
	}
	
	//      Use a normal approximation if P > PLIMIT
	if (p > plimit) {
		pn1 = three * sqrt(p) * (pow(x/p,one/three) + one / (nine * p) - one);
		return alnorm(pn1, false);
	}
	
	//      If X is extremely large compared to P then set gamma_prob = 1
	if (x > xbig) {
		return one;
	}
	
	if (x <= one || x < p) {
		//!      Use Pearson's series expansion.
		//!      (Note that P is not large enough to force overflow in LNGAMMA)
		
		arg = p * log(x) - x - lngamma(p + one);
		c = one;
		gamma_prob = one;
		a = p;
		do {
			a = a + one;
			c = c * x / a;
			gamma_prob = gamma_prob + c;
		} while (c >= tol);
		
		arg = arg + log(gamma_prob);
		gamma_prob = zero;
		if (arg >= elimit) {
			gamma_prob = exp(arg);
		} 
	} else {
		//!      Use a continued fraction expansion
		
		arg = p * log(x) - x - lngamma(p);
		a = one - p;
		b = a + x + one;
		c = zero;
		pn1 = one;
		pn2 = x;
		pn3 = x + one;
		pn4 = x * b;
		gamma_prob = pn3 / pn4;
		do {
			a = a + one;
			b = b + two;
			c = c + one;
			an = a * c;
			pn5 = b * pn3 - an * pn1;
			pn6 = b * pn4 - an * pn2;
			if (fabs(pn6) > zero) {
				rn = pn5 / pn6;
				if(fabs(gamma_prob - rn) <= MIN(tol, tol * rn))
					break;
				gamma_prob = rn;
			}
			
			pn1 = pn3;
			pn2 = pn4;
			pn3 = pn5;
			pn4 = pn6;
			if (fabs(pn5) >= oflo) {
				//  !      Re-scale terms in continued fraction if terms are large
				
				pn1 = pn1 / oflo;
				pn2 = pn2 / oflo;
				pn3 = pn3 / oflo;
				pn4 = pn4 / oflo;
			}
		} while (true);
		arg = arg + log(gamma_prob);
		gamma_prob = one;
		if (arg >= elimit) {
			gamma_prob = one - exp(arg);
		}
	}
	return gamma_prob;
}



double chi_squared(int ndf, double chi2) {
// Calculate the chi-squared distribution function
// ndf  = number of degrees of freedom
// chi2 = chi-squared value
// prob = probability of a chi-squared value <= chi2 (i.e. the left-hand
//        tail area)
	return gammad(0.5*chi2, 0.5*ndf);
}

void disaster() {
}


/*
The below function is from Alan Miller's package which has been made available
freely, and may be freely distributed.

The author is:
     Author: Alan Miller
             CSIRO Division of Mathematics & Statistics
             Private Bag 10, Rosebank MDC
             Clayton 3169, Victoria, Australia
     Phone: (+61) 3 9545-8036      Fax: (+61) 3 9545-8080
     e-mail: Alan.Miller @ mel.dms.csiro.au

The original code was translated from Fortran by Aleks Jakulin.     
*/
void logistic(// input
 			   int& ier,
			   int ngroups,			// # of examples
			   double **x,			// examples (real-valued)
			   int k,				// # attributes
			   double *s, double *n,// s-successes of n-trials
			   
			   // output
			   double& chisq,			// chi-squared
			   double& devnce,			// deviance
			   int& ndf,				// degrees of freedom
			   double *beta,			// fitted beta coefficients
			   double *se_beta,			// beta std.devs
			   double *fit,				// fitted probabilities for groups
			   double **cov_beta,		// approx covariance matrix
			   double *stdres,		 	// residuals
			   int *dependent
			   )			
{
	int i, iter, j, ncov, pos;
	double *propn, *p, *wt, *xrow, 
		*db, *bnew, dev_new, xb, *pnew, *covmat,
		*wnew, *range, var, *e, hii;
	bool *lindep;
	lsq fitter;
	
	ier = 0;
	ndf = ngroups - k - 1;
	if (ngroups < 2 || ndf < 0) {
		ier = 1; // not enough examples
		return;
	}
	for(i = 1; i <= ngroups; ++i) {
		if (s[i] < 0) {
			ier = 2;
			return;
		}
		if (n[i] < 0) {
			ier = 3;
			return;
		}
		if (n[i] < s[i]) {
			ier = 4;
			return;
		}
	}
	range = (double *)malloc(sizeof(double)*(k+1));
	for (i = 1; i <= k; ++i) {
		double min = INF;
		double max = -INF;
		
		for (j = 1; j <= ngroups; ++j) {
			if (x[j][i] < min)
				min = x[j][i];
			if (x[j][i] > max)
				max= x[j][i];
		}
		range[i] = max-min;
		if (range[i] < EPSILON*(fabs(min)+fabs(max))) {
			free(range);
			ier = 5; // constant variable
			//printf("variable %d is constant\n",i);
			dependent[i] = 1;
			return;
		}
	}
	
	++ngroups; ++k;
	propn = (double *)malloc(sizeof(double)*ngroups);
	p = (double *)malloc(sizeof(double)*ngroups);
	pnew = (double *)malloc(sizeof(double)*ngroups);
	wnew = (double *)malloc(sizeof(double)*ngroups);
	e = (double *)malloc(sizeof(double)*ngroups);
	wt = (double *)malloc(sizeof(double)*ngroups);
	
	xrow = (double *)malloc(sizeof(double)*k);
	db = (double *)malloc(sizeof(double)*k);	
	bnew = (double *)malloc(sizeof(double)*k);
	lindep = (bool *)malloc(sizeof(bool)*k);
	--k; --ngroups;
	
	for(i = 1; i <= ngroups; ++i) {
		/*
		printf("\n%2.2f %2.2f: ",s[i],n[i]);
		for (j = 1; j <= k; ++j) {
			printf("%5.2f ",x[i][j]);
		}*/
		propn[i] = s[i]/n[i];
		wt[i] = 1.0;
		p[i] = 0.5;
	}
	
	for(i = 0; i <= k; ++i) {
		beta[i] = 0.0;
	}
	
	iter = 1;
	
	do {
		fitter.startup(k,true);
		for (i = 1; i <= ngroups; ++i) {
			if (iter == 0) {
				xrow[0] = 0.25;
				for (j = 1; j <= k; ++j) {
					xrow[j] = 0.25*x[i][j];
				}
			} else {
				xrow[0] = p[i]*(1.0-p[i]);
				for (j = 1; j <= k; ++j) {
					xrow[j] = p[i]*(1.0-p[i])*x[i][j];
				}
			}
			fitter.includ(wt[i], xrow-1, propn[i]-p[i]);
		}
		
		//! Test for a singularity
		fitter.sing(lindep-1, ier);
		if (ier != 0) {
			for( i = 1; i<= k; ++i) {
				if (lindep[i]) {
					dependent[i] = 1;
					//printf("Variable number %d is linearly dependent upon earlier variables\n",i);
				}
			}
			ier = 6;
			return;
		}
		fitter.regcf(db-1, k+1, ier); // corrected
l10: 
		for (i = 0; i <= k; ++i) {
			bnew[i] = beta[i]+db[i];
		}
		
		//! Calculate new p(i)'s, weights & deviance
		
/*		double beta_sum=0.0;
		for (i=0; i<=k; i++) {
			beta_sum+=beta[i]*beta[i]*0.5;
		}
		dev_new = beta_sum; */

		dev_new = 0.0;
		for (i = 1; i <= ngroups; ++i) {
			xb = bnew[0];
			for (j = 1; j <= k; ++j) {
				xb += x[i][j] * bnew[j];
			}
			xb = exp(xb);
			pnew[i] = xb / (1.0 + xb);
			wnew[i] = (n[i])/(pnew[i]*(1.0 - pnew[i]));
			if (iter == 1) 
				wnew[i] = sqrt(wnew[i]);
			if (s[i] > 0) 
				dev_new = dev_new + s[i]*log(propn[i]/pnew[i]);
			if (s[i] < n[i]) 
				dev_new += (n[i]-s[i])*log((1.0-propn[i])/(1.0-pnew[i]));
		}
		dev_new = 2 * dev_new;
		//! If deviance has increased, reduce the step size.
		
		if (iter > 2 && dev_new > devnce*1.0001) {
			for (i = 0; i <= k; ++i)
				db[i] = 0.5 * db[i];
			goto l10;
		}
		//! Replace betas, weights & p's with new values
		
		for (i = 0; i <= k; ++i) {
			beta[i] = bnew[i];
		}
		for (i = 1; i <= ngroups; ++i) {
			wt[i] = wnew[i];
			p[i] = pnew[i];
		}
		//! Test for convergence
		
//		printf("iter. %d, dev: %f\n",iter,devnce-dev_new);
		if (iter > 2 && devnce - dev_new < 0.0001)
			break;
		devnce = dev_new;
		iter = iter + 1;
		if (iter > 200) {
			ier = 8;
			return;
		}
				
		//! Test for a very large beta
		
		for( i = 1; i<= k; ++i) {
			if(fabs(beta[i])*range[i] > 30.0) {
			//if(fabs(beta[i])*range[i] > 1e20) {	
				dependent[i] = 1;
//				printf("Coefficient for variable no. %d tending to infinity",i);
				
				ier = 7;
				return;
			}
		}
	} while(true);

	chisq = 0.0;
	for (i = 1; i <= ngroups; ++i) {
		double tt;

		e[i] = n[i]*p[i];
		tt = s[i]-e[i];
		chisq += tt*tt/e[i];
	}
	devnce = dev_new;

//! Calculate the approximate covariance matrix for the beta's, if ndf > 0.

	covmat = NULL;
	if (ndf > 0) {
		ncov = (k+1)*(k+2)/2;
		covmat = (double *)malloc(sizeof(double)*(ncov+1));
		fitter.cov(k+1, var, covmat, ncov, se_beta-1, ier);
		if (var < 1.0) {
			for (i = 1; i <= ncov; ++i) {
				covmat[i] /= var;
			}
			for (i = 0; i <= k; ++i) {
			    se_beta[i] /= sqrt(var);
			}
		}
		if (cov_beta != NULL) {
			pos = 1;
			for(i = 0;i<= k; ++i) {
				cov_beta[i][i] = covmat[pos];
				pos = pos + 1;
				for(j = i+1; j<= k; ++j) {
					cov_beta[i][j] = covmat[pos];
					cov_beta[j][i] = covmat[pos];
					pos = pos + 1;
				}
			}
		}
	}
	
	if (fit != NULL) {
		for (i = 1; i <= ngroups; ++i) {
			fit[i] = p[i];
		}
	}

	if (stdres != NULL) {
		for (i = 1; i <= ngroups; ++i) {
			xrow[0] = p[i]*(1.0 - p[i]);
			for (j = 1; j <= k; ++j) {
				xrow[j] = p[i]*(1.0 - p[i])*x[i][j];
			}
			fitter.hdiag(xrow-1, k+1, hii, ier);
			stdres[i] = (s[i]-n[i]*p[i]) / sqrt(n[i]*p[i]*(1.0-p[i])*(1.0-hii));
		}
	}

	if (covmat != NULL)
		free(covmat);
	free(propn); 
	free(p); 
	free(wnew); 
	free(pnew); 
	free(e); 
	free(wt);	
	free(xrow); 
	free(db); 
	free(bnew); 
	free(range); 
	free(lindep);
}
