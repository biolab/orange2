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


#include "cost.ppp"

void TCostMatrix::init(const float &inside)
{ 
  if (costs)
    delete costs;

  const int size = dimension * dimension;

  costs = new float[dimension * dimension];
  float *ci, *ce;
  for(ci = costs, ce = costs + size; ci != ce; *ci++ = inside);

  int dim = dimension;
  for(ci = costs; dim--; *ci = 0, ci += dimension+1);
}



TCostMatrix::TCostMatrix(const int &dim, const float &inside)
: dimension(dim),
  costs(NULL)
{
  if (dimension <= 0)
    raiseError("invalid dimension (%i)", dimension);

  init(inside);
}


TCostMatrix::TCostMatrix(PVariable acv, const float &inside)
: classVar(acv),
  dimension(0),
  costs(NULL)
{ 
  TEnumVariable *dcv = classVar.AS(TEnumVariable);
  if (!dcv)
    raiseError("attribute '%s' is not discrete", classVar->get_name().c_str());

  dimension = dcv->noOfValues();
  if (!dimension)
    raiseError("attribute '%s' has no values", classVar->get_name().c_str());

  init(inside);
}
