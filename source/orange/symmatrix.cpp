/*
    This file is part of Orange.

    Orange is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Orange is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Orange; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Authors: Janez Demsar, Blaz Zupan, 1996--2002
    Contact: janez.demsar@fri.uni-lj.si
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

