#if !defined(MATHUTIL_H)
#define MATHUTIL_H
  // singular value decomposition method for solving linear regression problem
  void svbksb(double **u, double w[], double **v, int m, int n, double b[],
      double x[]);
   void svdcmp(double **a, int m, int n, double w[], double **v);
   void svdfit(double x[], double y[], double sig[], int ndata, double a[], marray<int> &, int ma,
      double **u, double **v, double w[], double *chisq,
      void (*funcs)(double, double [], marray<int> &, int)) ;
   void svdvar(double **v, int ma, double w[], double **cvm) ;
   double pythag(double a, double b)  ;

 double dbrent(double ax, double bx, double cx, double (*f)(double),
   double (*df)(double), double tol, double *xmin) ;
 void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc,
   double (*func)(double)) ;
 void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),
   void (*dfunc)(double [], double [])) ;
 void frprmn(double p[], int n, double ftol, int *iter, double *fret,
   double (*func)(double []), void (*dfunc)(double [], double [])) ;
 double df1dim(double x) ;

 void powell(double p[], double **xi, marray<int> &, int n, double ftol, int *iter, double *fret,
   double (*func)(double [], marray<int> &)) ;

 void linmin(double p[], double xi[], int n, double *fret, double (*func)(double [],marray<int> &)) ;

 double brent(double ax, double bx, double cx, double (*f)(double), double tol,
   double *xmin) ;
 double f1dim(double x) ;

 double amotsa(double **p, double y[], double psum[], int ndim, double pb[],
   double *yb, double (*funk)(double []), int ihi, double *yhi, double fac)  ;
 void amebsa(double **p, double y[], int ndim, double pb[], double *yb, double ftol,
   double (*funk)(double []), int *iter, double temptr) ;
 double ran1(long *idum) ;

void lubksb(double **a, int n, int *indx, double b[]) ;
void ludcmp(double **a, int n, int *indx, double *d) ;


#endif
