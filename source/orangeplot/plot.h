#ifndef PLOT_H
#define PLOT_H

#include <QtGui/QGraphicsView>
#include <QtCore/QHash>
#include <QtCore/QMap>

#include "curve.h"

class Point;
class PlotItem;

class Plot : public QGraphicsView
{
    Q_OBJECT
public:
    enum SelectionBehavior
    {
        AddSelection,
        RemoveSelection,
        ToggleSelection
    };
    
    typedef QSet<DataPoint> PointSet;
    typedef QHash<DataPoint, Point*> PointHash;

    
    explicit Plot(QWidget* parent = 0);
    virtual ~Plot();
    
    virtual void replot() = 0;
    
    void add_item(PlotItem* item);
    void remove_item(PlotItem* item);
    
    QRectF data_rect_for_axes(int x_axis, int y_axis);
    QPair< double, double > bounds_for_axis(int axis);
    
    QList<PlotItem*> plot_items();
    
    void set_graph_rect(const QRectF rect);
    
    QGraphicsRectItem* graph_item;
    
    void set_dirty();
    
    void select_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void select_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    void mark_points(const QRectF& rect, SelectionBehavior behavior = AddSelection);
    void mark_points(const QPolygonF& area, SelectionBehavior behavior = AddSelection);
    
    QList< int > selected_points(const QList< double > x_data, const QList< double > y_data, const QTransform& transform);
    
    Point* point_at(const DataPoint& pos);
    Point* selected_point_at(const DataPoint& pos);
    
    void add_point(const DataPoint& pos, Point* item, PlotItem* parent);
    void add_points(const Data& data, const QList<Point*>& items, PlotItem* parent);
    void remove_point(const DataPoint& pos, PlotItem* parent);
    void remove_all_points(PlotItem* parent);
       
protected:
    void set_clean();
    bool is_dirty();
    
private:
    QList<PlotItem*> m_items;
    bool m_dirty;
    QGraphicsRectItem* clipItem;
    QMap<PlotItem*, PointSet> m_point_set;
    QMap<PlotItem*, PointHash> m_point_hash;
};

#endif // PLOT_H
