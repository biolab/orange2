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
