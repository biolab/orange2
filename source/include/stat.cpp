#include <functional>

#include "stat.hpp"

using namespace std;

int nUniquePoints(const vector<double> &points)
{ 
  int up = 0;
  for(vector<double>::const_iterator vb(points.begin()), vi(vb), ve(points.end()); vi!=ve; vi++)
    if ((vi == vb) || (*(vi-1) != *vi))
      up++;

  return up;
}


void samplingFactor(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  for (vector<double>::const_iterator pi(points.begin()), pe(points.end());;) {
    const double &ax = *pi;
    result.push_back(ax);
    
    if (++pi==pe)
      break;
    if (*pi == ax)
      continue;

    if (*pi != ax) {
      // We could write this faster, but we don't want to run into problems with rounding
      double div = (*pi - ax) / nPoints;
      for (int i=1; i < nPoints; i++)
        result.push_back(ax + i*div);
    }
  }
}


void samplingFactor(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  for (map<double, double>::const_iterator pi(points.begin()), pe(points.end());;) {
    const double &ax = (*pi).first;
    result.push_back(ax);
    
    if (++pi==pe)
      break;

    // We could write this faster, but we don't want to run into problems with rounding floats
    const double &div = ((*pi).first - ax) / nPoints;
    for (int i=1; i < nPoints; i++)
      result.push_back(ax + i*div);
  }
}


void samplingMinimal(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  if (nPoints<=points.size()) {
    for (map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
      result.push_back((*pi).first);
  }

  else
    samplingFixed(points, nPoints, result);
}


void samplingMinimal(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  int nUnique = nUniquePoints(points);

  if (nPoints<=nUnique)
    result = points;
  else
    samplingFixed(points, nPoints, result);
}


void samplingFixed(const vector<double> &points, int nPoints, vector<double> &result)
{ 
  int nUnique = nUniquePoints(points);

  result.clear();
  const double &ineach = double(nPoints - nUnique) / double(nUnique-1);
  double inthis = 0.0;
  
  for (vector<double>::const_iterator pi(points.begin()), pe(points.end());;) {
    double ax = *pi;
    result.push_back(ax);

    if (++pi==pe)
      break;

    if (*pi != ax) {
      inthis += ineach;
      if (inthis >= 1.0) {
        const double &dif = (*pi - ax) / (int(floor(inthis))+1);
        while (inthis > 0.5) {
          result.push_back(ax += dif);
          inthis -= 1.0;
        }
      }
    }
  }
}


void samplingFixed(const map<double, double> &points, int nPoints, vector<double> &result)
{ 
  result.clear();

  const double &ineach = float(nPoints - points.size()) / float(points.size()-1);
  double inthis = 0.0;
  
  for (map<double, double>::const_iterator pi(points.begin()), pe(points.end());;) {
    double ax = (*pi).first;
    result.push_back(ax);

    if (++pi==pe)
      break;

    inthis += ineach;
    if (inthis >= 0.5) {
      const double &dif = ((*pi).first - ax) / (int(floor(inthis))+1);
      while (inthis > 0.5) {
        result.push_back(ax += dif);
        inthis -= 1.0;
      }
    }
  }
}


void samplingUniform(const vector<double> &points, int nPoints, vector<double> &result)
{
  result.clear();

  const double &fi = points.front();
  const double &rg = (points.back()-fi) / (nPoints-1);
  for (int i = 0; i<nPoints; i++)
    result.push_back(fi + i*rg);
}


void samplingUniform(const map<double, double> &points, int nPoints, vector<double> &result)
{
  result.clear();

  const double &fi = (*points.begin()).first;
  map<double, double>::const_iterator pe(points.end());
  pe--;
  const double &rg = ((*pe).first-fi) / (nPoints-1);
  for (int i = 0; i<nPoints; i++)
    result.push_back(fi + i*rg);
}



bool comp1st(const pair<double, double> &x1, const pair<double, double> &x2)
{ return x1.first < x2.first; }


void vector2weighted(const vector<pair<double, double> > &points, vector<TXYW> &weighted)
{
  if (points.empty())
    throw StatException("lwr/loess: empty sample");

  weighted.clear();

  vector<pair<double, double> > myPoints = points;
  sort(myPoints.begin(), myPoints.end(), comp1st);

  vector<pair<double, double> >::const_iterator mpi(myPoints.begin()), mpe(myPoints.end());
  weighted.push_back(TXYW((*mpi).first, (*mpi).second));
  while(++mpi != mpe) {
    TXYW &last = weighted.back();
    if ((*mpi).first == last.x) {
      last.y += (*mpi).second;
      last.w += 1.0;
    }
    else {
      if (last.w > 1e-6)
        last.y /= last.w;
      weighted.push_back(TXYW((*mpi).first, (*mpi).second));
    }
  }

  TXYW &last = weighted.back();
  if (last.w > 1e-6)
    last.y /= last.w;
}

void loess(const vector<double> &refpoints, const vector<TXYW> &points, const float &windowProp, vector<pair<double, double> > &result)
{ 
  result.clear();
  
  typedef vector<TXYW>::const_iterator iterator;

  iterator lowedge = points.begin();
  iterator highedge = points.end();
  iterator from;
  iterator to;

  double nPoints = 0;
  for(from = points.begin(); from != highedge; nPoints += (*(from++)).w);
  double needpoints = windowProp <= 1.0 ? nPoints * windowProp : windowProp;

  bool stopWindow = needpoints >= nPoints;
  if (stopWindow) {
    from = lowedge;
    to = highedge;
  }
  else
    for(from = to = lowedge; (to != highedge) && (needpoints>0); needpoints -= (*(to++)).w);

  for(vector<double>::const_iterator rpi(refpoints.begin()), rpe(refpoints.end()); rpi != rpe; rpi++) {

    const double &refx = *rpi;

    /* Adjust the window */

    if (!stopWindow) {
      // adjust the top end so that the window includes the reference point
      //   (note that the last point included is to-1, so this one must be >= refx)
      for(; (to != highedge) && (refx > (*(to-1)).x); needpoints -= (*(to++)).w);
      // adjust the bottom end as high as it goes but so that the window still covers at least needpoints points
      for(; (from != to) && ((*to).x - refx < refx - (*from).x) && (needpoints + (*from).w < 0); needpoints += (*(from++)).w);

      while ((to!=highedge) && ((*to).x - refx < refx - (*from).x)) {
        // 'to' is not at the high edge and to's point is closer that from's, so include it
        needpoints -= (*(to++)).w;
        // adjust the bottom end as high as it goes but so that the window still covers at least needpoints points
        for(; (from != to) && (needpoints + (*from).w < 0); needpoints += (*(from++)).w);
      }

      stopWindow = (to==highedge);
    }

 
    /* Determine the window half-width */
    double h = abs(refx - (*from).x);
    const double h2 = abs((*(to-1)).x - refx);
    if (h2 > h)
      h = h2;

    h *= 1.1;


    /* Iterate through the window */

    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0, Sw = 0.0, Swx = 0.0, Swxx = 0.0;
    double n = 0.0;

    for (iterator ii = from; ii != to; ii++) {
      const double &x = (*ii).x;
      const double &y = (*ii).y;

      // compute the weight based on the distance
      double w = abs(refx - x) / h;
      w = 1 - w*w*w;
      w = w*w*w;
      // and multiply it by the point's given weight
      w *= (*ii).w;

      n   += w;
      Sx  += w * x;
      Sxx += w * x * x;
      Sy  += w * y;
      Syy += w * y * y;
      Sxy += w * x * y;

      Sw  += w * w;
      Swx += w * w * x;
      Swxx += w * w * x * x;
    }

    if (n==0) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double mu_x = Sx / n;
    const double mu_y = Sy / n;
    const double sigma_x2 = (Sxx - mu_x * Sx) / n;
    if (sigma_x2 < 1e-20) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double sigma_y2 = (Syy - mu_y * Sy) / n;
    const double sigma_xy = (Sxy - Sx * Sy / n) / n;
    const double sigma_y_x = sigma_y2 - sigma_xy * sigma_xy / sigma_x2;

    const double dist_x = refx - mu_x;
    const double y = mu_y + sigma_xy / sigma_x2 * dist_x;

    double var_y = sigma_y_x / n / n * (Sw + dist_x * dist_x / sigma_x2 / sigma_x2 * (Swxx + mu_x * mu_x * Sw - 2 * mu_x * Swx));
    if ((var_y < 0) && (var_y > -1e-6))
      var_y = 0;
    else
      var_y = sqrt(var_y);

    result.push_back(pair<double, double>(y, var_y));
  }
}


void loess(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &windowProp, vector<pair<double, double> > &result)
{
  vector<TXYW> weighted;
  vector2weighted(points, weighted);
  loess(refpoints, weighted, windowProp, result);
}


void loess(const vector<double> &refpoints, const map<double, double> &points, const float &windowProp, vector<pair<double, double> > &result)
{
  vector<TXYW> opoints;
  for(map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
    opoints.push_back(TXYW((*pi).first, (*pi).second));

  loess(refpoints, opoints, windowProp, result);
}


void lwr(const vector<double> &refpoints, const vector<TXYW> &points, const float &smoothFactor, vector<pair<double, double> > &result)
{ 
  result.clear();
  
  typedef vector<TXYW>::const_iterator iterator;

  float tot_w = 0.0;
  { 
    const_ITERATE(vector<TXYW>, pi, points)
    tot_w += (*pi).w;
  }
  const float p25 = 0.25 * tot_w;
  const float p75 = 0.75 * tot_w;
  float x25, x75;
  tot_w = 0;
  { 
    vector<TXYW>::const_iterator pi(points.begin()), pe(points.end());
    for(; (pi!=pe) && (tot_w<p25); tot_w += (*pi).w, pi++);
    const float &x1 = (*(pi-1)).x;
    x25 = x1 + ((*pi).x-x1) * (p25 - tot_w + (*pi).w) / (*pi).w;

    if (tot_w >= p75)
      throw StatException("not enough data to compute 25th and 75th percentile");

    for(; (pi!=pe) && (tot_w<p75); tot_w += (*pi).w, pi++);
    const float &x2 = (*(pi-1)).x;
    x75 = x2 + ((*pi).x-x2) * (p75 - tot_w + (*pi).w) / (*pi).w;
  }

  const float sigma = smoothFactor * (x75-x25);

  const_ITERATE(vector<double>, ri, refpoints) {
    const double &refx = *ri;

    double Sx = 0.0, Sy = 0.0, Sxx = 0.0, Syy = 0.0, Sxy = 0.0, Sw = 0.0, Swx = 0.0, Swxx = 0.0;
    double n = 0.0;

    for (vector<TXYW>::const_iterator ii(points.begin()), ie(points.end()); ii != ie; ii++) {
      const double &x = (*ii).x;
      const double &y = (*ii).y;

      // compute the weight based on the distance and the point's given weight
      const double dx = x - *ri;
      double w = (*ii).w * exp(- dx*dx / (sigma*sigma));

      n   += w;
      Sx  += w * x;
      Sxx += w * x * x;
      Sy  += w * y;
      Syy += w * y * y;
      Sxy += w * x * y;

      Sw  += w * w;
      Swx += w * w * x;
      Swxx += w * w * x * x;
    }

    if (n==0) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double mu_x = Sx / n;
    const double mu_y = Sy / n;
    const double sigma_x2 = (Sxx - mu_x * Sx) / n;
    if (sigma_x2 < 1e-20) {
      result.push_back(pair<double, double>(Sy, 0));
      continue;
    }

    const double sigma_y2 = (Syy - mu_y * Sy) / n;
    const double sigma_xy = (Sxy - Sx * Sy / n) / n;
    const double sigma_y_x = sigma_y2 - sigma_xy * sigma_xy / sigma_x2;

    const double dist_x = refx - mu_x;
    const double y = mu_y + sigma_xy / sigma_x2 * dist_x;

    double var_y = sigma_y_x / n / n * (Sw + dist_x * dist_x / sigma_x2 / sigma_x2 * (Swxx + mu_x * mu_x * Sw - 2 * mu_x * Swx));
    if ((var_y < 0) && (var_y > -1e-6))
      var_y = 0;
    else
      var_y = sqrt(var_y);

    result.push_back(pair<double, double>(y, var_y));
  }
}



void lwr(const vector<double> &refpoints, const vector<pair<double, double> > &points, const float &smoothFactor, vector<pair<double, double> > &result)
{
  vector<TXYW> weighted;
  vector2weighted(points, weighted);
  lwr(refpoints, weighted, smoothFactor, result);
}


void lwr(const vector<double> &refpoints, const map<double, double> &points, const float &smoothFactor, vector<pair<double, double> > &result)
{
  vector<TXYW> opoints;
  for(map<double, double>::const_iterator pi(points.begin()), pe(points.end()); pi != pe; pi++)
    opoints.push_back(TXYW((*pi).first, (*pi).second));

  lwr(refpoints, opoints, smoothFactor, result);
}




double alnorm(double x, bool upper) {
	double norm_prob, con=1.28, z,y, ltone = 7.0, utzero = 18.66;
	double p, r, a2, b1, c1, c3, c5, d1, d3, d5;
	double q, a1, a3, b2, c2, c4, c6, d2, d4;
	bool up;
	
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

double PPND(double P,int& IER) {
	//C
	//C ALGORITHM AS 111, APPL.STATIST., VOL.26, 118-121, 1977.
	//C
	//C PRODUCES NORMAL DEVIATE CORRESPONDING TO LOWER TAIL AREA = P.
	//C
	//C	See also AS 241 which contains alternative routines accurate to
	//C	about 7 and 16 decimal digits.
	//C
	double SPLIT = 0.42;
	double A[] = {2.50662823884,-18.61500062529,41.39119773534,-25.44106049637};
	double B[] = {-8.47351093090,23.08336743743,-21.06224101826,3.13082909833};
	double C[] = {-2.78718931138,-2.29796479134,4.85014127135,2.32121276858};
	double D[] = {3.54388924762,1.63706781897};
	double ZERO = 0.0, ONE = 1.0, HALF = 0.5;
	double temp, Q, R;
	
	IER = 0;
	Q = P-HALF;
	if (fabs(Q) <= SPLIT) {
		//C
		//C 0.08 < P < 0.92
		//C
		R = Q*Q;
		return Q*(((A[3]*R + A[2])*R + A[1])*R + A[0])/((((B[3]*R + B[2])*R + B[1])*R + B[0])*R + ONE);
	} else {
		//C
		//C P < 0.08 OR P > 0.92, SET R = MIN(P,1-P)
		//C
		R = P;
		if (Q > ZERO) 
			R = ONE-P;
		if (R <= ZERO) {
			IER = 1;
			return ZERO;
		}
		R = sqrt(-log(R));
		temp = (((C[3]*R + C[2])*R + C[1])*R + C[0])/((D[1]*R + D[0])*R + ONE);
		if (Q < ZERO)
			return -temp;
		else
			return temp;
	}
}

double POLY(double *c, int nord, double x) {
	//c
	//c
	//C Algorithm AS 181.2   Appl. Statist.  (1982) Vol. 31, No. 2
	//c
	//C Calculates the algebraic polynomial of order nored-1 with
	//C array of coefficients c.  Zero order coefficient is c(1)
	//c
	double temp, p;
	int j,i,n2;
	
    temp = c[0];
    if(nord == 1)
		return temp;
    p = x*c[nord-1];
    if(nord != 2) {
		n2 = nord-2;
		j = n2;
		for (i = 0; i < n2; ++i) {
			p = (p+c[j])*x;
			--j;
		}
    }
    return temp + p;
}

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define SIGN(a) ((a) > 0.0 ? (1) : (-1))

// X: SORTED data
// n: q. of data (3-5000)
// A: weights?

// does not work when all x identical, or when the range too small
// n2 = n % 2
// R call: init=false, X, n, n,n2,single[n2], w, pw, ifault)
double SWILK(bool INIT, double *X, int N, int N1, int N2, double *A, double& W, double &PW, int& IFAULT) {
	//
	//C
	//C ALGORITHM AS R94 APPL. STATIST. (1995) VOL.44, NO.4
	//C
	//C Calculates the Shapiro-Wilk W test and its significance level
	//C
	
	double C1[6] = {0.0E0, 0.221157E0, -0.147981E0, -0.207119E1, 0.4434685E1, -0.2706056E1};
	double C2[6] = {0.0E0, 0.42981E-1, -0.293762E0, -0.1752461E1, 0.5682633E1, -0.3582633E1};
	double C3[4] = {0.5440E0, -0.39978E0, 0.25054E-1, -0.6714E-3};
	double C4[4] = {0.13822E1, -0.77857E0, 0.62767E-1, -0.20322E-2};
	double C5[4] = {-0.15861E1, -0.31082E0, -0.83751E-1, 0.38915E-2};
	double C6[3] = {-0.4803E0, -0.82676E-1, 0.30302E-2};
	double C7[2] = {0.164E0, 0.533E0};
	double C8[2] = {0.1736E0, 0.315E0};
	double C9[2] = {0.256E0, -0.635E-2};
	double G[2] =  {-0.2273E1, 0.459E0};
	double Z90 = 0.12816E1, Z95 = 0.16449E1, Z99 = 0.23263E1;
	double ZM = 0.17509E1, ZSS = 0.56268E0;
	double ZERO = 0.0, ONE = 1.0, TWO = 2.0;
	double BF1 = 0.8378E0, XX90 = 0.556E0, XX95 = 0.622E0;
	double THREE = 3.0, SQRTH = 0.70711E0;
	double QTR = 0.25E0, TH = 0.375E0, SMALL = 1E-19;
	double PI6 = 0.1909859E1, STQR = 0.1047198E1;
	
	double SUMM2, SSUMM2, FAC, RSN, AN, AN25, A1, A2, DELTA, RANGE;
	double SA, SX, SSX, SSA, SAX, ASA, XSX, SSASSX, W1, Y, XX, XI;
	double GAMMA, M, S, LD, BF, Z90F, Z95F, Z99F, ZFM, ZSD, ZBAR;
	
	int NCENS, NN2, I, I1, J;
	bool UPPER = true;
	
	PW  =  ONE;
	if (W > ZERO) 
		W = ONE;
	AN = N;
	IFAULT = 3;
	NN2 = N/2;
	if (N2 < NN2)
		return PW;
	IFAULT = 1;
	if (N < 3)
		return PW;
	
	//C If INIT is false, calculates coefficients for the test
	if (!INIT) {
		if (N == 3)
			A[0] = SQRTH;
		else {
			AN25 = AN + QTR;
			SUMM2 = ZERO;
			for(I = 0; I < N2; ++I) {
				int itmp;
				A[I] = PPND((I + 1 - TH)/AN25,itmp);
				SUMM2 += A[I]*A[I];
			}
			SUMM2 *= TWO;
			SSUMM2 = sqrt(SUMM2);
			RSN = ONE / sqrt(AN);
			A1 = POLY(C1, 6, RSN) - A[0] / SSUMM2;
		}
		//C
		//C Normalize coefficients
		//C
		if (N > 5) {
			I1 = 3;
			A2 = -A[1]/SSUMM2 + POLY(C2,6,RSN);
			FAC = sqrt((SUMM2 - TWO * A[0]*A[0] - TWO * A[1]*A[1])/(ONE - TWO * A1*A1 - TWO * A2*A2));
			A[0] = A1;
			A[1] = A2;
		} else {
			I1 = 2;
			FAC = sqrt((SUMM2 - TWO * A[0]*A[0])/(ONE - TWO * A1*A1));
			A[0] = A1;
		}
		for (I = I1; I <= NN2; ++I)
			A[I-1] = -A[I-1]/FAC;
		INIT = true;
	}
	
	if (N1 < 3)
		return PW;
	NCENS = N - N1;
	IFAULT = 4;
	if (NCENS < 0 || (NCENS > 0 && N < 20))
		return PW;
	IFAULT = 5;
	DELTA = float(NCENS)/AN;
	if (DELTA > 0.8)
		return PW;
	//C
	//C If W input as negative, calculate significance level of -W
	//C
	
	if (W < ZERO) {
		W1 = ONE + W;
		IFAULT = 0;
	} else {
		//C
		//C Check for zero range
		//C
		IFAULT = 6;
		RANGE = X[N1-1] - X[0];
		if (RANGE < SMALL)
			return PW;
		//C
		//C Check for correct sort order on range - scaled X
		//C
		IFAULT = 7;
		XX = X[0]/RANGE;
		SX = XX;
		SA = -A[0];
		J = N;
		for (I = 2; I <= N1; ++I) {
			XI = X[I-1]/RANGE;
			if (XX-XI > SMALL)
				return PW; // FIXED BY JD, WAS: IFAULT=7;
			SX += XI;
			if (I != J) 
				SA += SIGN(I - J) * A[MIN(I, J)-1];
			XX = XI;
			--J;
		}
		IFAULT = 0;
		if (N > 5000) 
			IFAULT = 2;
		//C Calculate W statistic as squared correlation
		//C between data and coefficients
		SA = SA/N1;
		SX = SX/N1;
		SSA = ZERO;
		SSX = ZERO;
		SAX = ZERO;
		J = N;
		for (I = 1; I <= N1; ++I) {
			if (I != J) 
				ASA = SIGN(I - J) * A[MIN(I, J)-1] - SA;
			else
				ASA = -SA;
			
			XSX = X[I-1]/RANGE - SX;
			SSA += ASA * ASA;
			SSX += XSX * XSX;
			SAX +=  ASA * XSX;
			--J;
		}
		
		//C W1 equals (1-W) claculated to avoid excessive rounding error
		//C for W very near 1 (a potential problem in very large samples)
		SSASSX = sqrt(SSA * SSX);
		W1 = (SSASSX - SAX) * (SSASSX + SAX)/(SSA * SSX);
	}
	
	W = ONE - W1;
	//C
	//C Calculate significance level for W (exact for N=3)
	//C
	if (N == 3) {
		PW = PI6 * (asin(sqrt(W)) - STQR);
		return PW;
	}
	Y = log(W1);
	XX = log(AN);
	M = ZERO;
	S = ONE;
	if (N <= 11) {
		GAMMA = POLY(G, 2, AN);
		if (Y >= GAMMA) {
			PW = SMALL;
			return PW;
		}
		Y = -log(GAMMA - Y);
		M = POLY(C3, 4, AN);
		S = exp(POLY(C4, 4, AN));
	} else {
		M = POLY(C5, 4, XX);
		S = exp(POLY(C6, 3, XX));
	}
	if (NCENS > 0) {
		//C
		//C Censoring by proportion NCENS/N.  Calculate mean and sd
		//C of normal equivalent deviate of W.
		//C
		LD = -log(DELTA);
		BF = ONE + XX * BF1;
		Z90F = Z90 + BF * pow(POLY(C7, 2, pow(XX90,XX)), LD);
		Z95F = Z95 + BF * pow(POLY(C8, 2, pow(XX95,XX)), LD);
		Z99F = Z99 + BF * pow(POLY(C9, 2, XX),LD);
		//C
		//C Regress Z90F,...,Z99F on normal deviates Z90,...,Z99 to get
		//C pseudo-mean and pseudo-sd of z as the slope and intercept
		//C
		ZFM = (Z90F + Z95F + Z99F)/THREE;
		ZSD = (Z90*(Z90F-ZFM)+Z95*(Z95F-ZFM)+Z99*(Z99F-ZFM))/ZSS;
		ZBAR = ZFM - ZSD * ZM;
		M += ZBAR * S;
		S *= ZSD;
	}
	PW = alnorm((Y - M)/S, UPPER);
	return PW;
}
