#ifndef PLOTITEM_H
#define PLOTITEM_H

#include <QtGui/QGraphicsItem>

class Graph;

class PlotItem : public QGraphicsItem
{

public:
    PlotItem(QList<double> xData, QList<double> yData, QGraphicsItem* parent = 0, QGraphicsScene* scene = 0);
    virtual ~PlotItem();
    
    virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QRectF boundingRect() const;
    
    virtual QRectF dataRect() const;
    
    void attach(Graph* graph);
    void detach();
    
    static QRectF boundingRectFromData(QList<double> xData, QList<double> yData);
    
    bool isAutoScale() const;
    void setAutoScale(bool autoScale);
    
    QPair<int, int> axes() const;
    void setAxes(int x_axis, int y_axis);
    
private:
    Q_DISABLE_COPY(PlotItem)
    
    Graph* m_graph;
    QRectF m_dataRect;
    QPair<int, int> m_axes;
    bool m_autoScale;
    
    friend class Graph;
    
};

#endif // PLOTITEM_H
