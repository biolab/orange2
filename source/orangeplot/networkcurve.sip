%ModuleCode
#include "networkcurve.h"
%End

%MappedType Coord /DocType="tuple-of-float-float"/
{
%TypeHeaderCode
#include "networkcurve.h"
%End

%ConvertFromTypeCode
    // Create the tuple.
    return Py_BuildValue((char *)"dd", sipCpp->first, sipCpp->second);
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PyTuple_Size(sipPy) == 2);
    
    Coord* qp = new Coord;

    PyArg_ParseTuple(sipPy, "dd", &qp->first, &qp->second);
    *sipCppPtr = qp;
    
    return sipGetState(sipTransferObj);
%End
};

class NetworkCurve : Curve
{

%TypeHeaderCode
#include "networkcurve.h"
%End

public:
    typedef QMap<int, Coord> Coordinates;
    typedef QList<QPair<int, int> > Edges;
    
    NetworkCurve(const NetworkCurve::Coordinates& coordinates, const NetworkCurve::Edges& edges, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~NetworkCurve();
    
    virtual void updateProperties();
    
    NetworkCurve::Edges edges;
    NetworkCurve::Coordinates coordinates;
};