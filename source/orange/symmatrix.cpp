#include "symmatrix.ppp"

int TSymMatrix::getindex(const int &i, const int &j, bool raiseExceptions) const
{
  if (i==j) {
    if ((i>=dim) || (i<0))
      raiseError("index out of range");
    return (i*(i+3))>>1;
  }

  if (i>j) {
    if ((i>=dim) || (j<0))
      raiseError("index out of range");
    if ((matrixType == Upper) || (matrixType == UpperFilled))
      if (raiseExceptions)
        raiseError("index out of range for upper triangular matrix");
      else
        return -1;
    return ((i*(i+1))>>1) + j;
  }

  if ((j>=dim) || (i<0))
    raiseError("index out of range");
  if ((matrixType == Lower) || (matrixType == LowerFilled))
    if (raiseExceptions)
      raiseError("index out of range for lower triangular matrix");
    else
      return -1;
  return  ((j*(j+1))>>1) + i;
}

