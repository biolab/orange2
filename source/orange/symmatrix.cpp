/*
    This file is part of Orange.
    
    Copyright 1996-2010 Faculty of Computer and Information Science, University of Ljubljana
    Contact: janez.demsar@fri.uni-lj.si

    Orange is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange.  If not, see <http://www.gnu.org/licenses/>.
*/

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

