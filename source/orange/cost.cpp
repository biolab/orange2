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


#include "cost.ppp"

void TCostMatrix::init(const int &dimension, const float &inside)
{ 
  reserve(dimension);
  for(int i = 0; i<dimension; i++) {
    push_back(mlnew TFloatList(dimension, inside));
    back()->operator[](i) = 0.0;
  }
}



TCostMatrix::TCostMatrix(const int &dimension, const float &inside)
{
  if (dimension <= 0)
    raiseError("invalid dimension (%i)", dimension);

  init(dimension, inside);
}


TCostMatrix::TCostMatrix(PVariable acv, const float &inside)
: classVar(acv)
{ 
  TEnumVariable *dcv = classVar.AS(TEnumVariable);
  if (!dcv)
    raiseError("attribute '%s' is not discrete", classVar->name.c_str());

  const int dimension = dcv->noOfValues();
  if (!dimension)
    raiseError("attribute '%s' has no values", classVar->name.c_str());

  init(dcv->noOfValues(), inside);
}
