
#define NRANSI
#include <math.h>
#include "nrutil.h"
#include "contain.h"
#include "mathutil.h"


 // singular value decomposition method for solving linear regression problem
#define TOL 1.0e-5

// mask is aded to assure that only selected parameters are fitted (those with value 1),
// those with value 0 will stay unchanged
void svdfit(double x[], double y[], double sig[], int ndata, double A[], marray<int> &mask, int ma,
   double **u, double **v, double w[], double *chisq,
   void (*funcs)(double, double [], marray<int> &,int))
{

   int maskCounter = 1;
   int j,i;
   double wmax,tmp,thresh,sum,*b,*afunc;
   b=vector(1,ndata);
   afunc=vector(1,ma);
   double *maskedA = vector(1, ma) ;
   for (i=1 ; i < mask.len() ; i++)
     if (mask[i] == 1)
       maskedA[maskCounter++] = A[i] ;

   for (i=1;i<=ndata;i++) {
      (*funcs)(x[i],afunc,mask,ma);
      tmp=1.0/sig[i];
      for (j=1;j<=ma;j++) u[i][j]=afunc[j]*tmp;
      b[i]=y[i]*tmp;
   }
   svdcmp(u,ndata,ma,w,v);
   wmax=0.0;
   for (j=1;j<=ma;j++)
      if (w[j] > wmax) wmax=w[j];
   thresh=TOL*wmax;
   for (j=1;j<=ma;j++)
      if (w[j] < thresh) w[j]=0.0;
   svbksb(u,w,v,ndata,ma,b,maskedA);
   *chisq=0.0;
   for (i=1;i<=ndata;i++) {
      (*funcs)(x[i],afunc,mask,ma);
      for (sum=0.0,j=1;j<=ma;j++) sum += maskedA[j]*afunc[j];
      *chisq += (tmp=(y[i]-sum)/sig[i],tmp*tmp);
   }

   maskCounter = 1 ;
   for (i=1 ; i < mask.len() ; i++)
     if (mask[i] == 1)
       A[i] = maskedA[maskCounter++] ;

   free_vector(maskedA, 1, ma) ;
   free_vector(afunc,1,ma);
   free_vector(b,1,ndata);
}
#undef TOL


void svdvar(double **v, int ma, double w[], double **cvm)
{
   int k,j,i;
   double sum,*wti;

   wti=vector(1,ma);
   for (i=1;i<=ma;i++) {
      wti[i]=0.0;
      if (w[i]) wti[i]=1.0/(w[i]*w[i]);
   }
   for (i=1;i<=ma;i++) {
      for (j=1;j<=i;j++) {
         for (sum=0.0,k=1;k<=ma;k++) sum += v[i][k]*v[j][k]*wti[k];
         cvm[j][i]=cvm[i][j]=sum;
      }
   }
   free_vector(wti,1,ma);
}


void svbksb(double **u, double w[], double **v, int m, int n, double b[], double x[])
{
   int jj,j,i;
   double s,*tmp;

   tmp=vector(1,n);
   for (j=1;j<=n;j++) {
      s=0.0;
      if (w[j]) {
         for (i=1;i<=m;i++) s += u[i][j]*b[i];
         s /= w[j];
      }
      tmp[j]=s;
   }
   for (j=1;j<=n;j++) {
      s=0.0;
      for (jj=1;jj<=n;jj++) s += v[j][jj]*tmp[jj];
      x[j]=s;
   }
   free_vector(tmp,1,n);
}


void svdcmp(double **a, int m, int n, double w[], double **v)
{
   double pythag(double a, double b);
   int flag,i,its,j,jj,k,l,nm;
   double anorm,c,f,g,h,s,scale,x,y,z,*rv1;

   rv1=vector(1,n);
   g=scale=anorm=0.0;
   for (i=1;i<=n;i++) {
      l=i+1;
      rv1[i]=scale*g;
      g=s=scale=0.0;
      if (i <= m) {
         for (k=i;k<=m;k++) scale += fabs(a[k][i]);
         if (scale) {
            for (k=i;k<=m;k++) {
               a[k][i] /= scale;
               s += a[k][i]*a[k][i];
            }
            f=a[i][i];
            g = -SIGN(sqrt(s),f);
            h=f*g-s;
            a[i][i]=f-g;
            for (j=l;j<=n;j++) {
               for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
               f=s/h;
               for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
            }
            for (k=i;k<=m;k++) a[k][i] *= scale;
         }
      }
      w[i]=scale *g;
      g=s=scale=0.0;
      if (i <= m && i != n) {
         for (k=l;k<=n;k++) scale += fabs(a[i][k]);
         if (scale) {
            for (k=l;k<=n;k++) {
               a[i][k] /= scale;
               s += a[i][k]*a[i][k];
            }
            f=a[i][l];
            g = -SIGN(sqrt(s),f);
            h=f*g-s;
            a[i][l]=f-g;
            for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
            for (j=l;j<=m;j++) {
               for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
               for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
            }
            for (k=l;k<=n;k++) a[i][k] *= scale;
         }
      }
      anorm=FMAX(anorm,(fabs(w[i])+fabs(rv1[i])));
   }
   for (i=n;i>=1;i--) {
      if (i < n) {
         if (g) {
            for (j=l;j<=n;j++)
               v[j][i]=(a[i][j]/a[i][l])/g;
            for (j=l;j<=n;j++) {
               for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
               for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
            }
         }
         for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
      }
      v[i][i]=1.0;
      g=rv1[i];
      l=i;
   }
   for (i=IMIN(m,n);i>=1;i--) {
      l=i+1;
      g=w[i];
      for (j=l;j<=n;j++) a[i][j]=0.0;
      if (g) {
         g=1.0/g;
         for (j=l;j<=n;j++) {
            for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
            f=(s/a[i][i])*g;
            for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
         }
         for (j=i;j<=m;j++) a[j][i] *= g;
      } else for (j=i;j<=m;j++) a[j][i]=0.0;
      ++a[i][i];
   }
   for (k=n;k>=1;k--) {
      for (its=1;its<=30;its++) {
         flag=1;
         for (l=k;l>=1;l--) {
            nm=l-1;
            if ((double)(fabs(rv1[l])+anorm) == anorm) {
               flag=0;
               break;
            }
            if ((double)(fabs(w[nm])+anorm) == anorm) break;
         }
         if (flag) {
            c=0.0;
            s=1.0;
            for (i=l;i<=k;i++) {
               f=s*rv1[i];
               rv1[i]=c*rv1[i];
               if ((double)(fabs(f)+anorm) == anorm) break;
               g=w[i];
               h=pythag(f,g);
               w[i]=h;
               h=1.0/h;
               c=g*h;
               s = -f*h;
               for (j=1;j<=m;j++) {
                  y=a[j][nm];
                  z=a[j][i];
                  a[j][nm]=y*c+z*s;
                  a[j][i]=z*c-y*s;
               }
            }
         }
         z=w[k];
         if (l == k) {
            if (z < 0.0) {
               w[k] = -z;
               for (j=1;j<=n;j++) v[j][k] = -v[j][k];
            }
            break;
         }
         if (its == 30) nrerror("no convergence in 30 svdcmp iterations");
         x=w[l];
         nm=k-1;
         y=w[nm];
         g=rv1[nm];
         h=rv1[k];
         f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
         g=pythag(f,1.0);
         f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
         c=s=1.0;
         for (j=l;j<=nm;j++) {
            i=j+1;
            g=rv1[i];
            y=w[i];
            h=s*g;
            g=c*g;
            z=pythag(f,h);
            rv1[j]=z;
            c=f/z;
            s=h/z;
            f=x*c+g*s;
            g = g*c-x*s;
            h=y*s;
            y *= c;
            for (jj=1;jj<=n;jj++) {
               x=v[jj][j];
               z=v[jj][i];
               v[jj][j]=x*c+z*s;
               v[jj][i]=z*c-x*s;
            }
            z=pythag(f,h);
            w[j]=z;
            if (z) {
               z=1.0/z;
               c=f*z;
               s=h*z;
            }
            f=c*g+s*y;
            x=c*y-s*g;
            for (jj=1;jj<=m;jj++) {
               y=a[jj][j];
               z=a[jj][i];
               a[jj][j]=y*c+z*s;
               a[jj][i]=z*c-y*s;
            }
         }
         rv1[l]=0.0;
         rv1[k]=f;
         w[k]=x;
      }
   }
   free_vector(rv1,1,n);
}


double pythag(double a, double b)
{
   double absa,absb;
   absa=fabs(a);
   absb=fabs(b);
   if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
   else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}



/*
#define ITMAX 200
#define EPS 1.0e-10
#define FREEALL free_vector(xi,1,n);free_vector(h,1,n);free_vector(g,1,n);

void frprmn(double p[], int n, double ftol, int *iter, double *fret,
   double (*func)(double []), void (*dfunc)(double [], double []))
{
   void linmin(double p[], double xi[], int n, double *fret,
      double (*func)(double []));
   int j,its;
   double gg,gam,fp,dgg;
   double *g,*h,*xi;

   g=vector(1,n);
   h=vector(1,n);
   xi=vector(1,n);
   fp=(*func)(p);
   (*dfunc)(p,xi);
   for (j=1;j<=n;j++) {
      g[j] = -xi[j];
      xi[j]=h[j]=g[j];
   }
   for (its=1;its<=ITMAX;its++) {
      *iter=its;
      linmin(p,xi,n,fret,func);
      if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
         FREEALL
         return;
      }
      fp=(*func)(p);
      (*dfunc)(p,xi);
      dgg=gg=0.0;
      for (j=1;j<=n;j++) {
         gg += g[j]*g[j];
         dgg += (xi[j]+g[j])*xi[j];
      }
      if (gg == 0.0) {
         FREEALL
         return;
      }
      gam=dgg/gg;
      for (j=1;j<=n;j++) {
         g[j] = -xi[j];
         xi[j]=h[j]=g[j]+gam*h[j];
      }
   }
   nrerror("Too many iterations in frprmn");
}
#undef ITMAX
#undef EPS
#undef FREEALL
*/

#define TOL 2.0e-4
int ncom;
double *pcom,*xicom,(*nrfunc)(double [],marray<int>&);
void (*nrdfun)(double [], double []);

/*
void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),
   void (*dfunc)(double [], double []))
{
   double dbrent(double ax, double bx, double cx,
      double (*f)(double), double (*df)(double), double tol, double *xmin);
   double f1dim(double x);
   double df1dim(double x);
   void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
      double *fc, double (*func)(double));
   int j;
   double xx,xmin,fx,fb,fa,bx,ax;

   ncom=n;
   pcom=vector(1,n);
   xicom=vector(1,n);
   nrfunc=func;
   nrdfun=dfunc;
   for (j=1;j<=n;j++) {
      pcom[j]=p[j];
      xicom[j]=xi[j];
   }
   ax=0.0;
   xx=1.0;
   mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
   *fret=dbrent(ax,xx,bx,f1dim,df1dim,TOL,&xmin);
   for (j=1;j<=n;j++) {
      xi[j] *= xmin;
      p[j] += xi[j];
   }
   free_vector(xicom,1,n);
   free_vector(pcom,1,n);
}
#undef TOL
#undef NRANSI
*/

#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc,
   double (*func)(double))
{
   double ulim,u,r,q,fu,dum;

   *fa=(*func)(*ax);
   *fb=(*func)(*bx);
   if (*fb > *fa) {
      SHFT(dum,*ax,*bx,dum)
      SHFT(dum,*fb,*fa,dum)
   }
   *cx=(*bx)+GOLD*(*bx-*ax);
   *fc=(*func)(*cx);
   while (*fb > *fc) {
      r=(*bx-*ax)*(*fb-*fc);
      q=(*bx-*cx)*(*fb-*fa);
      u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
         (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
      ulim=(*bx)+GLIMIT*(*cx-*bx);
      if ((*bx-u)*(u-*cx) > 0.0) {
         fu=(*func)(u);
         if (fu < *fc) {
            *ax=(*bx);
            *bx=u;
            *fa=(*fb);
            *fb=fu;
            return;
         } else if (fu > *fb) {
            *cx=u;
            *fc=fu;
            return;
         }
         u=(*cx)+GOLD*(*cx-*bx);
         fu=(*func)(u);
      } else if ((*cx-u)*(u-ulim) > 0.0) {
         fu=(*func)(u);
         if (fu < *fc) {
            SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
            SHFT(*fb,*fc,fu,(*func)(u))
         }
      } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
         u=ulim;
         fu=(*func)(u);
      } else {
         u=(*cx)+GOLD*(*cx-*bx);
         fu=(*func)(u);
      }
      SHFT(*ax,*bx,*cx,u)
      SHFT(*fa,*fb,*fc,fu)
   }
}
#undef GOLD
#undef GLIMIT
#undef TINY
#undef SHFT

#define ITMAX 100
#define ZEPS 1.0e-10
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);

double dbrent(double ax, double bx, double cx, double (*f)(double),
   double (*df)(double), double tol, double *xmin)
{
   int iter,ok1,ok2;
   double a,b,d=0.0,d1,d2,du,dv,dw,dx,e=0.0;
   double fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;

   a=(ax < cx ? ax : cx);
   b=(ax > cx ? ax : cx);
   x=w=v=bx;
   fw=fv=fx=(*f)(x);
   dw=dv=dx=(*df)(x);
   for (iter=1;iter<=ITMAX;iter++) {
      xm=0.5*(a+b);
      tol1=tol*fabs(x)+ZEPS;
      tol2=2.0*tol1;
      if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
         *xmin=x;
         return fx;
      }
      if (fabs(e) > tol1) {
         d1=2.0*(b-a);
         d2=d1;
         if (dw != dx) d1=(w-x)*dx/(dx-dw);
         if (dv != dx) d2=(v-x)*dx/(dx-dv);
         u1=x+d1;
         u2=x+d2;
         ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
         ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
         olde=e;
         e=d;
         if (ok1 || ok2) {
            if (ok1 && ok2)
               d=(fabs(d1) < fabs(d2) ? d1 : d2);
            else if (ok1)
               d=d1;
            else
               d=d2;
            if (fabs(d) <= fabs(0.5*olde)) {
               u=x+d;
               if (u-a < tol2 || b-u < tol2)
                  d=SIGN(tol1,xm-x);
            } else {
               d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
            }
         } else {
            d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
         }
      } else {
         d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
      }
      if (fabs(d) >= tol1) {
         u=x+d;
         fu=(*f)(u);
      } else {
         u=x+SIGN(tol1,d);
         fu=(*f)(u);
         if (fu > fx) {
            *xmin=x;
            return fx;
         }
      }
      du=(*df)(u);
      if (fu <= fx) {
         if (u >= x) a=x; else b=x;
         MOV3(v,fv,dv, w,fw,dw)
         MOV3(w,fw,dw, x,fx,dx)
         MOV3(x,fx,dx, u,fu,du)
      } else {
         if (u < x) a=u; else b=u;
         if (fu <= fw || w == x) {
            MOV3(v,fv,dv, w,fw,dw)
            MOV3(w,fw,dw, u,fu,du)
         } else if (fu < fv || v == x || v == w) {
            MOV3(v,fv,dv, u,fu,du)
         }
      }
   }
   nrerror("Too many iterations in routine dbrent");
   return 0.0;
}
#undef ITMAX
#undef ZEPS
#undef MOV3


//extern int ncom;
//extern double *pcom,*xicom,(*nrfunc)(double []);
//extern void (*nrdfun)(double [], double []);

double df1dim(double x)
{
   int j;
   double df1=0.0;
   double *xt,*df;

   xt=vector(1,ncom);
   df=vector(1,ncom);
   for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
   (*nrdfun)(xt,df);
   for (j=1;j<=ncom;j++) df1 += df[j]*xicom[j];
   free_vector(df,1,ncom);
   free_vector(xt,1,ncom);
   return df1;
}

#define ITMAX 200

marray<int> nrMask ;

void powell(double p[], double **xi,marray<int> &Mask, int n, double ftol, int *iter, double *fret,
   double (*func)(double [], marray<int> &Mask))
{
   void linmin(double p[], double xi[], int n, double *fret,
      double (*func)(double [], marray<int>& Mask));
   int i,ibig,j;
   double del,fp,fptt,t,*pt,*ptt,*xit;

   pt=vector(1,n);
   ptt=vector(1,n);
   xit=vector(1,n);
   *fret=(*func)(p,Mask);
   nrMask.copy(Mask) ;
   for (j=1;j<=n;j++) pt[j]=p[j];
   for (*iter=1;;++(*iter)) {
      fp=(*fret);
      ibig=0;
      del=0.0;
      for (i=1;i<=n;i++) {
         for (j=1;j<=n;j++) xit[j]=xi[j][i];
         fptt=(*fret);
         linmin(p,xit,n,fret,func);
         if (fabs(fptt-(*fret)) > del) {
            del=fabs(fptt-(*fret));
            ibig=i;
         }
      }
      if (2.0*fabs(fp-(*fret)) <= ftol*(fabs(fp)+fabs(*fret))) {
         free_vector(xit,1,n);
         free_vector(ptt,1,n);
         free_vector(pt,1,n);
         return;
      }
      if (*iter == ITMAX) nrerror("powell exceeding maximum iterations.");
      for (j=1;j<=n;j++) {
         ptt[j]=2.0*p[j]-pt[j];
         xit[j]=p[j]-pt[j];
         pt[j]=p[j];
      }
      fptt=(*func)(ptt,Mask);
      if (fptt < fp) {
         t=2.0*(fp-2.0*(*fret)+fptt)*SQR(fp-(*fret)-del)-del*SQR(fp-fptt);
         if (t < 0.0) {
            linmin(p,xit,n,fret,func);
            for (j=1;j<=n;j++) {
               xi[j][ibig]=xi[j][n];
               xi[j][n]=xit[j];
            }
         }
      }
   }
}
#undef ITMAX


#define TOL 2.0e-4


void linmin(double p[], double xi[], int n, double *fret, double (*func)(double [], marray<int> &Mask))
{
   double brent(double ax, double bx, double cx,
      double (*f)(double), double tol, double *xmin);
   double f1dim(double x);
   void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,
      double *fc, double (*func)(double));
   int j;
   double xx,xmin,fx,fb,fa,bx,ax;

   ncom=n;
   pcom=vector(1,n);
   xicom=vector(1,n);
   nrfunc=func;
   for (j=1;j<=n;j++) {
      pcom[j]=p[j];
      xicom[j]=xi[j];
   }
   ax=0.0;
   xx=1.0;
   mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,f1dim);
   *fret=brent(ax,xx,bx,f1dim,TOL,&xmin);
   for (j=1;j<=n;j++) {
      xi[j] *= xmin;
      p[j] += xi[j];
   }
   free_vector(xicom,1,n);
   free_vector(pcom,1,n);
}
#undef TOL

double f1dim(double x)
{
   int j;
   double f,*xt;

   xt=vector(1,ncom);
   for (j=1;j<=ncom;j++) xt[j]=pcom[j]+x*xicom[j];
   f=(*nrfunc)(xt,nrMask);
   free_vector(xt,1,ncom);
   return f;
}


#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);

double brent(double ax, double bx, double cx, double (*f)(double), double tol,
   double *xmin)
{
   int iter;
   double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
   double e=0.0;

   a=(ax < cx ? ax : cx);
   b=(ax > cx ? ax : cx);
   x=w=v=bx;
   fw=fv=fx=(*f)(x);
   for (iter=1;iter<=ITMAX;iter++) {
      xm=0.5*(a+b);
      tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
      if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
         *xmin=x;
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
      fu=(*f)(u);
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
   nrerror("Too many iterations in brent");
   *xmin=x;
   return fx;
}
#undef ITMAX
#undef CGOLD
#undef ZEPS
#undef SHFT

#define GET_PSUM \
               for (n=1;n<=ndim;n++) {\
               for (sum=0.0,m=1;m<=mpts;m++) sum += p[m][n];\
               psum[n]=sum;}
long idum;
double tt;

void amebsa(double **p, double y[], int ndim, double pb[], double *yb, double ftol,
   double (*funk)(double []), int *iter, double temptr)
{
   double amotsa(double **p, double y[], double psum[], int ndim, double pb[],
      double *yb, double (*funk)(double []), int ihi, double *yhi, double fac);
   double ran1(long *idum);
   int i,ihi,ilo,j,m,n,mpts=ndim+1;
   double rtol,sum,swap,yhi,ylo,ynhi,ysave,yt,ytry,*psum;

   psum=vector(1,ndim);
   tt = -temptr;
   GET_PSUM
   for (;;) {
      ilo=1;
      ihi=2;
      ynhi=ylo=y[1]+tt*log(ran1(&idum));
      yhi=y[2]+tt*log(ran1(&idum));
      if (ylo > yhi) {
         ihi=1;
         ilo=2;
         ynhi=yhi;
         yhi=ylo;
         ylo=ynhi;
      }
      for (i=3;i<=mpts;i++) {
         yt=y[i]+tt*log(ran1(&idum));
         if (yt <= ylo) {
            ilo=i;
            ylo=yt;
         }
         if (yt > yhi) {
            ynhi=yhi;
            ihi=i;
            yhi=yt;
         } else if (yt > ynhi) {
            ynhi=yt;
         }
      }
      rtol=2.0*fabs(yhi-ylo)/(fabs(yhi)+fabs(ylo));
      if (rtol < ftol || *iter < 0) {
         swap=y[1];
         y[1]=y[ilo];
         y[ilo]=swap;
         for (n=1;n<=ndim;n++) {
            swap=p[1][n];
            p[1][n]=p[ilo][n];
            p[ilo][n]=swap;
         }
         break;
      }
      *iter -= 2;
      ytry=amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,-1.0);
      if (ytry <= ylo) {
         ytry=amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,2.0);
      } else if (ytry >= ynhi) {
         ysave=yhi;
         ytry=amotsa(p,y,psum,ndim,pb,yb,funk,ihi,&yhi,0.5);
         if (ytry >= ysave) {
            for (i=1;i<=mpts;i++) {
               if (i != ilo) {
                  for (j=1;j<=ndim;j++) {
                                         psum[j]=0.5*(p[i][j]+p[ilo][j]);
                                         p[i][j]=psum[j];
                  }
                  y[i]=(*funk)(psum);
               }
            }
            *iter -= ndim;
            GET_PSUM
         }
      } else ++(*iter);
   }
   free_vector(psum,1,ndim);
}
#undef GET_PSUM


double amotsa(double **p, double y[], double psum[], int ndim, double pb[],
   double *yb, double (*funk)(double []), int ihi, double *yhi, double fac)
{
   double ran1(long *idum);
   int j;
   double fac1,fac2,yflu,ytry,*ptry;

   ptry=vector(1,ndim);
   fac1=(1.0-fac)/ndim;
   fac2=fac1-fac;
   for (j=1;j<=ndim;j++)
      ptry[j]=psum[j]*fac1-p[ihi][j]*fac2;
   ytry=(*funk)(ptry);
   if (ytry <= *yb) {
      for (j=1;j<=ndim;j++) pb[j]=ptry[j];
      *yb=ytry;
   }
   yflu=ytry-tt*log(ran1(&idum));
   if (yflu < *yhi) {
      y[ihi]=ytry;
      *yhi=yflu;
      for (j=1;j<=ndim;j++) {
         psum[j] += ptry[j]-p[ihi][j];
         p[ihi][j]=ptry[j];
      }
   }
   free_vector(ptry,1,ndim);
   return yflu;
}



#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran1(long *idum)
{
   int j;
   long k;
   static long iy=0;
   static long iv[NTAB];
   double temp;

   if (*idum <= 0 || !iy) {
      if (-(*idum) < 1) *idum=1;
      else *idum = -(*idum);
      for (j=NTAB+7;j>=0;j--) {
         k=(*idum)/IQ;
         *idum=IA*(*idum-k*IQ)-IR*k;
         if (*idum < 0) *idum += IM;
         if (j < NTAB) iv[j] = *idum;
      }
      iy=iv[0];
   }
   k=(*idum)/IQ;
   *idum=IA*(*idum-k*IQ)-IR*k;
   if (*idum < 0) *idum += IM;
   j=iy/NDIV;
   iy=iv[j];
   iv[j] = *idum;
   if ((temp=AM*iy) > RNMX) return RNMX;
   else return temp;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX


#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software '>'!^,. */

void lubksb(double **a, int n, int *indx, double b[])
{
	int i,ii=0,ip,j;
	double sum;

	for (i=1;i<=n;i++) {
		ip=indx[i];
		sum=b[ip];
		b[ip]=b[i];
		if (ii)
			for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
		else if (sum) ii=i;
		b[i]=sum;
	}
	for (i=n;i>=1;i--) {
		sum=b[i];
		for (j=i+1;j<=n;j++) sum -= a[i][j]*b[j];
		b[i]=sum/a[i][i];
	}
}

#define TINY 1.0e-20;

void ludcmp(double **a, int n, int *indx, double *d)
{
	int i,imax,j,k;
	double big,dum,sum,temp;
	double *vv;

	vv=vector(1,n);
	*d=1.0;
	for (i=1;i<=n;i++) {
		big=0.0;
		for (j=1;j<=n;j++)
			if ((temp=fabs(a[i][j])) > big) big=temp;
		if (big == 0.0) nrerror("Singular matrix in routine ludcmp");
		vv[i]=1.0/big;
	}
	for (j=1;j<=n;j++) {
		for (i=1;i<j;i++) {
			sum=a[i][j];
			for (k=1;k<i;k++) sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
		}
		big=0.0;
		for (i=j;i<=n;i++) {
			sum=a[i][j];
			for (k=1;k<j;k++)
				sum -= a[i][k]*a[k][j];
			a[i][j]=sum;
			if ( (dum=vv[i]*fabs(sum)) >= big) {
				big=dum;
				imax=i;
			}
		}
		if (j != imax) {
			for (k=1;k<=n;k++) {
				dum=a[imax][k];
				a[imax][k]=a[j][k];
				a[j][k]=dum;
			}
			*d = -(*d);
			vv[imax]=vv[j];
		}
		indx[j]=imax;
		if (a[j][j] == 0.0) a[j][j]=TINY;
		if (j != n) {
			dum=1.0/(a[j][j]);
			for (i=j+1;i<=n;i++) a[i][j] *= dum;
		}
	}
	free_vector(vv,1,n);
}
#undef TINY

