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
    enum SelectionBehavior
    {
        AddSelection,
        RemoveSelection,
        ToggleSelection,
        ReplaceSelection
    };
    
    Plot(QWidget* parent /TransferThis/ = 0);
    virtual ~Plot();
    
    virtual void replot() = 0;
    
    void add_item(PlotItem* item /Transfer/);
    void remove_item(PlotItem* item /TransferBack/);
    
    QRectF data_rect_for_axes(int xAxis, int yAxis);
    QPair< double, double > bounds_for_axis(int axis);
    
    QList<PlotItem*> plot_items();
    void set_graph_rect(const QRectF rect);

    void set_dirty();
    
    QGraphicsRectItem* graph_item;
    
    void select_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void select_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void unselect_all_points();
    
    void mark_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void mark_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void unmark_all_points();
    
    QList< int > selected_points(const QList< double > x_data, const QList< double > y_data, const QTransform& transform);
    QList< Point* > selected_points();
    Point* nearest_point(const QPointF& pos);
    Point* point_at(const DataPoint& pos);
    
Q_SIGNALS:
    void selection_changed();
    
protected:
    void set_clean();
    bool is_dirty();
};
