// QPair<int, int> is implemented as a Python 2-element tuple.
%MappedType QPair<double, double> /DocType="tuple-of-float-float"/
{
%TypeHeaderCode
#include <QtCore/QPair>
%End

%ConvertFromTypeCode
    // Create the tuple.
    return Py_BuildValue((char *)"dd", sipCpp->first, sipCpp->second);
%End

%ConvertToTypeCode
    // Check the type if that is all that is required.
    if (sipIsErr == NULL)
        return (PyTuple_Size(sipPy) == 2);
    
    QPair<double,double>* qp = new QPair<double,double>();

    PyArg_ParseTuple(sipPy, "dd", &qp->first, &qp->second);
    *sipCppPtr = qp;
    
    return sipGetState(sipTransferObj);
%End
};

class Graph : QGraphicsView {

%TypeHeaderCode
#include "graph.h"
%End

public:
    Graph(QWidget* parent /TransferThis/ = 0);
    virtual ~Graph();
    
    void addItem(PlotItem* item);
    void removeItem(PlotItem* item);
    void removeAllItems();
    
    QRectF dataRectForAxes(int xAxis, int yAxis);
    QPair< double, double > boundsForAxis(int axis);
    
    QList<PlotItem*> itemList();
    
    QGraphicsRectItem* graph_item;
};
