#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QtGui/QGraphicsItem>

class Plot;

class PlotItem : public QGraphicsItem
{

public:
    PlotItem(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF data_rect() const;
    void set_data_rect(const QRectF& dataRect);
    
    virtual void set_graph_transform(const QTransform& transform);
    virtual QTransform graph_transform() const;
    
    void attach(Plot* graph);
    void detach();
    Plot* plot();
    virtual void register_points();
    
    static QRectF rect_from_data(const QList<double>& x_data, const QList<double>& y_data);
    
    bool is_auto_scale() const;
    void set_auto_scale(bool auto_scale);
    
    QPair<int, int> axes() const;
    void set_axes(int x_axis, int y_axis);
    
    inline void set_x_axis(int x_axis)
    {
        set_axes(x_axis, axes().second);
    }
    inline void set_y_axis(int y_axis)
    {
        set_axes(axes().first, y_axis);
    }
    
private:
    Q_DISABLE_COPY(PlotItem)
    
    Plot* m_plot;
    QRectF m_dataRect;
    QPair<int, int> m_axes;
    bool m_autoScale;
    QTransform m_graphTransform;
    
    friend class Plot;
    
};

#endif // PLOTITEM_H
