#ifndef _GSL_IMPORTS_HPP
#define _GSL_IMPORTS_HPP

void * (*i__gsl_vector_alloc) (const size_t n);
void * (*i__gsl_vector_free) (void * m);
double (*i__gsl_vector_get) (const void * v, const size_t i);
void   (*i__gsl_vector_set) (void * v, const size_t i, double x);


void * (*i__gsl_matrix_alloc) (const size_t n1, const size_t n2);
void   (*i__gsl_matrix_free) (void * m);
double (*i__gsl_matrix_get)(const void * m, const size_t i, const size_t j);
void   (*i__gsl_matrix_set)(void * m, const size_t i, const size_t j, const double x);


void * (*i__gsl_multifit_linear_alloc) (size_t n, size_t p);
void   (*i__gsl_multifit_linear_free)(void * work);

int *  (*i__gsl_multifit_linear) (const void * X, const void * y, void * c, void * cov, double * chisq, void * work);
int *  (*i__gsl_multifit_wlinear) (const void * X, const void * w, const void * y, void * c, void * cov, double * chisq, void * work);


#define gsl_vector_get (*i__gsl_vector_get)
#define gsl_vector_set (*i__gsl_vector_set)
#define gsl_vector_alloc (*i__gsl_vector_alloc)
#define gsl_vector_free (*i__gsl_vector_free)

#define gsl_matrix_alloc (*i__gsl_matrix_alloc)
#define gsl_matrix_free (*i__gsl_matrix_free)
#define gsl_matrix_get (*i__gsl_matrix_get)
#define gsl_matrix_set (*i__gsl_matrix_set)

#define gsl_multifit_linear_alloc (*i__gsl_multifit_linear_alloc)
#define gsl_multifit_linear_free (*i__gsl_multifit_linear_free)

#define gsl_multifit_linear (*i__gsl_multifit_linear)
#define gsl_multifit_wlinear (*i__gsl_multifit_wlinear)

#endif