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

class Plot : QGraphicsView {

%TypeHeaderCode
#include "plot.h"
%End

public:
    Plot(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot();
    
    void add_item(PlotItem* item /Transfer/);
    void remove_item(PlotItem* item /TransferBack/);
    
    QRectF data_rect_for_axes(int xAxis, int yAxis);
    QPair< double, double > bounds_for_axis(int axis);
    
    QList<PlotItem*> plot_items();
    void set_graph_rect(const QRectF rect);

    void set_dirty();
    
    QGraphicsRectItem* graph_item;
    
protected:
    void set_clean();
    bool is_dirty();
};
