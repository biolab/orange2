#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QtGui/QGraphicsItem>

class Graph;

class PlotItem : public QGraphicsItem
{

public:
    PlotItem(QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF dataRect() const;
    void setDataRect(const QRectF& dataRect);
    
    virtual void setGraphTransform(const QTransform& transform);
    virtual QTransform graphTransform() const;
    
    void attach(Graph* graph);
    void detach();
    
    static QRectF boundingRectFromData(const QList<double>& xData, const QList<double>& yData);
    
    bool isAutoScale() const;
    void setAutoScale(bool autoScale);
    
    QPair<int, int> axes() const;
    void setAxes(int x_axis, int y_axis);
    
    inline void setXAxis(int x_axis)
    {
        setAxes(x_axis, axes().second);
    }
    inline void setYAxis(int y_axis)
    {
        setAxes(axes().first, y_axis);
    }
    
private:
    Q_DISABLE_COPY(PlotItem)
    
    Graph* m_graph;
    QRectF m_dataRect;
    QPair<int, int> m_axes;
    bool m_autoScale;
    QTransform m_graphTransform;
    
    friend class Graph;
    
};

#endif // PLOTITEM_H
