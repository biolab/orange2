#include <stdio.h>
#include <float.h>

#define NRANSI
#include "ftree.h"
#include "estimator.h"
#include "utils.h"
#include "nrutil.h"
#include "mathutil.h"
#include "rndforest.h"
#include "rfUtil.h"
#include "options.h"

extern Options *opt ;




//************************************************************
//
//                      rfRegularize
//                      ------------
//
//     computes regularization coefficients of random forest
//
//************************************************************

void featureTree::rfRegularize() {
   marray<double> a(opt->rfNoTrees+1, 0.0) ;
   int iter=-1 ;
   double fret = -1.0 ;
   rfRegFrprmn(opt->rfRegLambda, a, iter, fret) ;
   for (int i=0 ; i < opt->rfNoTrees; ++i)
      rfA[i] = a[i+1] ;
}

#define ITMAX 200
#define EPS 1.0e-10
double regLambda ;

void featureTree::rfRegFrprmn(double lambda, marray<double> &p, int &iter, double &fret) {
   int n = opt->rfNoTrees ;
   double ftol = 0.0001 ;
   regLambda = lambda ;
   int j,its;
   double gg,gam,fp,dgg;
   // double *g,*h,*xi;
   marray<double> xi(n+1), g(n+1), h(n+1);
   rfA0 = rfEvalA0() ;
   fp=rfRegEval(p, xi);
   for (j=1;j<=n;j++) {
      g[j] = -xi[j];
      xi[j]=h[j]=g[j];
   }
   for (its=1;its<=ITMAX;its++) {
      iter=its;
      rfLinmin(p,xi,n,fret);
      if (2.0*fabs(fret-fp) <= ftol*(fabs(fret)+fabs(fp)+EPS)) {
         return;
      }
      fp=rfRegEval(p, xi);
      dgg=gg=0.0;
      for (j=1;j<=n;j++) {
         gg += g[j]*g[j];
         dgg += (xi[j]+g[j])*xi[j];
      }
      if (gg == 0.0) {
         return;
      }
      gam=dgg/gg;
      for (j=1;j<=n;j++) {
         g[j] = -xi[j];
         xi[j]=h[j]=g[j]+gam*h[j];
      }
   }
   error("featureTree::rfRegFrprmn", "Too many iterations ");
   p.init(1.0/n) ;
}



#define TOL 2.0e-4
extern int ncom;
extern double* pcom, *xicom ;

double featureTree::f1dim(double x)
{
	int j;
	double f;
	marray<double> xt(ncom+1);

	for (j=1;j<=ncom;j++) 
	   xt[j]=pcom[j]+x*xicom[j];
	f=rfFunc(xt);
	return f;
}



void featureTree::rfLinmin(marray<double> &p, marray<double> &xi, int n, double &fret) {
	int j;
	double xx,xmin,fx,fb,fa,bx,ax;

	ncom=n;
    pcom=vector(1,n);
    xicom=vector(1,n);
	//pcom.create(n+1);
	//xicom.create(n+1);
	for (j=1;j<=n;j++) {
		pcom[j]=p[j];
		xicom[j]=xi[j];
	}
	ax=0.0;
	xx=1.0;
	rfmnbrak(ax,xx,bx,fa,fx,fb);
	fret=rfBrent(ax,xx,bx,TOL,xmin);
	for (j=1;j<=n;j++) {
		xi[j] *= xmin;
		p[j] += xi[j];
	}
   free_vector(xicom,1,n);
   free_vector(pcom,1,n);

}
#undef TOL
#undef ITMAX
#undef EPS
#undef FREEALL

    

// for 2 class problems, classes are considered 1 and -1
double featureTree::rfRegEval(marray<double> &a, marray<double> &g){
   marray<double> distr(NoClasses+1) ;
   int iT, i, max, oobN ;
   marray<double> oobF(NoTeachCases, 0.0) ;
   g.init(0.0) ;
   double Loss = 0.0, y, residium, r = 0.0 ;
   for (i=0 ; i < NoTeachCases ; i++) {
     oobN = 0 ;
     for (iT = 0 ; iT < opt->rfNoTrees ; iT++) {
		if (forest[iT].oob[i]) {
		    oobN++ ;
			max = rfTreeCheck(forest[iT].t.root, DTeach[i], distr) ;
            if (max==1)
			  oobF[i] += a[iT+1];
			else
			  oobF[i] -= a[iT+1];
		}
	 }
	 oobF[i] += rfA0 ;
	 if (oobN)
       oobF[i] /= double(oobN) ;
	 if (DiscData(DTeach[i],0) == 1) 
		 y = 1.0 ;
	  else 
		 y = -1.0 ;
	  residium = sqr(y - Mmax(-1.0, Mmin(1.0, oobF[i]))) ;
	  if (fabs(oobF[i]) < 1.0)
		 r= residium ;
	  else r = 0.0 ;
	  Loss  += sqr(residium) ;
      for (iT = 0 ; iT < opt->rfNoTrees ; iT++) {
		if (forest[iT].oob[i]) {
			max = rfTreeCheck(forest[iT].t.root, DTeach[i], distr) ;
			if (max == 1)
			   g[iT+1] += r ;
			else g[iT+1] -= r ;
		}
	 }

   }
   Loss /= double(NoTeachCases) ;
   double reg = 0.0 ;
   for (iT = 1 ; iT <= opt->rfNoTrees ; iT++) {
       reg += fabs(a[iT]) ;
       g[iT] *= 2.0/double(NoTeachCases) ;
	   if (a[iT] > 0)
		  g[iT] += regLambda ;
	   else if (a[iT] < 0)
		  g[iT] -= regLambda ;
   }
   double value = Loss + regLambda * reg;
   return value ;
}

// for 2 class problems, classes are considered 1 and -1
double featureTree::rfFunc(marray<double> &a){
   marray<double> distr(NoClasses+1) ;
   int iT, i, max, oobN ;
   marray<double> oobF(NoTeachCases, 0.0) ;
   double Loss = 0.0, y, residium ;
   for (i=0 ; i < NoTeachCases ; i++) {
     oobN = 0 ;
     for (iT = 0 ; iT < opt->rfNoTrees ; iT++) {
		if (forest[iT].oob[i]) {
		    oobN++ ;
			max = rfTreeCheck(forest[iT].t.root, DTeach[i], distr) ;
            if (max==1)
			  oobF[i] += a[iT+1];
			else
			  oobF[i] -= a[iT+1];
		}
	 }
	 oobF[i]+=rfA0 ;
	 if (oobN)
       oobF[i] /= double(oobN) ;
     if (DiscData(DTeach[i],0) == 1)
		 y = 1.0 ;
	  else 
		 y = -1.0 ;
	  residium = sqr(y - Mmax(-1.0, Mmin(1.0, oobF[i]))) ;
	  Loss  += sqr(residium) ;
   }
   Loss /= double(NoTeachCases) ;
   double reg = 0.0 ;
   for (iT = 1 ; iT <= opt->rfNoTrees ; iT++) {
       reg += fabs(a[iT]) ;
   }
   double value = Loss + regLambda * reg;
   return value ;
}


double featureTree::rfEvalA0(void){
   double def = 0.0  ;
   for (int i=0 ; i < NoTeachCases ; i++) {
	 if (DiscData(DTeach[i],0) == 1) 
		 def += 1.0 ;
	  else 
		 def -= -1.0 ;
   }
   return def / double(NoTeachCases) ;
}




#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

void featureTree::rfmnbrak(double &ax, double &bx, double &cx, double &fa, double &fb, double &fc)
{
	double ulim,u,r,q,fu,dum;

	fa=f1dim(ax);
	fb=f1dim(bx);
	if (fb > fa) {
		SHFT(dum,ax,bx,dum)
		SHFT(dum,fb,fa,dum)
	}
	cx=(bx)+GOLD*(bx-ax);
	fc=f1dim(cx);
	while (fb > fc) {
		r=(bx-ax)*(fb-fc);
		q=(bx-cx)*(fb-fa);
		u=(bx)-((bx-cx)*q-(bx-ax)*r)/
			(2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
		ulim=(bx)+GLIMIT*(cx-bx);
		if ((bx-u)*(u-cx) > 0.0) {
			fu=f1dim(u);
			if (fu < fc) {
				ax=(bx);
				bx=u;
				fa=(fb);
				fb=fu;
				return;
			} else if (fu > fb) {
				cx=u;
				fc=fu;
				return;
			}
			u=(cx)+GOLD*(cx-bx);
			fu=f1dim(u);
		} else if ((cx-u)*(u-ulim) > 0.0) {
			fu=f1dim(u);
			if (fu < fc) {
				SHFT(bx,cx,u,cx+GOLD*(cx-bx))
				SHFT(fb,fc,fu,f1dim(u))
			}
		} else if ((u-ulim)*(ulim-cx) >= 0.0) {
			u=ulim;
			fu=f1dim(u);
		} else {
			u=(cx)+GOLD*(cx-bx);
			fu=f1dim(u);
		}
		SHFT(ax,bx,cx,u)
		SHFT(fa,fb,fc,fu)
	}
}
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT


#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

double featureTree::rfBrent(double ax, double bx, double cx, double tol, double &xmin)
{
	int iter;
	double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
	double e=0.0;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=f1dim(x);
	for (iter=1;iter<=ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			r=(x-w)*(fx-fv);
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;
			q=2.0*(q-r);
			if (q > 0.0) p = -p;
			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
				d=CGOLD*(e=(x >= xm ? a-x : b-x));
			else {
				d=p/q;
				u=x+d;
				if (u-a < tol2 || b-u < tol2)
					d=SIGN(tol1,xm-x);
			}
		} else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));
		}
		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
		fu=f1dim(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			SHFT(v,w,x,u)
			SHFT(fv,fw,fx,fu)
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			} else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		}
	}
	error("featureTree::rfBrent", "Too many iterations");
	xmin=x;
	return fx;
}
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT
