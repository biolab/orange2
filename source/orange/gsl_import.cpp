#include "gsl_imports.hpp"

#include "jit_linker.hpp"

TJitLink gslLinks[] = {
{ (void **)&i__gsl_vector_get, "gsl_vector_get"},
{ (void **)&i__gsl_vector_set, "gsl_vector_set"},
{ (void **)&i__gsl_vector_alloc, "gsl_vector_alloc"},
{ (void **)&i__gsl_matrix_alloc, "gsl_matrix_alloc"},
{ (void **)&i__gsl_vector_free, "gsl_vector_free"},
{ (void **)&i__gsl_matrix_free, "gsl_matrix_free"},
{ (void **)&i__gsl_multifit_linear_alloc, "gsl_multifit_linear_alloc"},
{ (void **)&i__gsl_multifit_linear_free, "gsl_multifit_linear_alloc"},
{ (void **)&i__gsl_multifit_wlinear, "gsl_multifit_wlinear"},
{ (void **)&i__gsl_multifit_linear, "gsl_multifit_linear"}
};


