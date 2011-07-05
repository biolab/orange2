#include "plotitem.h"
#include "graph.h"

PlotItem::PlotItem(QGraphicsItem* parent, QGraphicsScene* scene): QGraphicsItem(parent, scene), 
m_graph(0)
{
    setAutoScale(true);
}

PlotItem::PlotItem(const PlotItem& ): QGraphicsItem(), 
m_graph(0)
{
    // Disabled copy constructor
}

PlotItem::~PlotItem()
{

}

void PlotItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{

}

QRectF PlotItem::boundingRect() const
{
    return QRectF();
}

void PlotItem::attach(Graph* graph)
{
    graph->addItem(this);
}

void PlotItem::detach()
{
    m_graph->removeItem(this);
}

QPair< int, int > PlotItem::axes() const
{
    return m_axes;
}

void PlotItem::setAxes(int x_axis, int y_axis)
{
    m_axes.first = x_axis;
    m_axes.second = y_axis;
}

bool PlotItem::isAutoScale() const
{
    return m_autoScale;
}

void PlotItem::setAutoScale(bool autoScale)
{
    m_autoScale = autoScale;
}

QRectF PlotItem::dataRect() const
{
    return m_dataRect;
}

void PlotItem::setGraphTransform(const QTransform& transform)
{
    m_graphTransform = transform;
}

QTransform PlotItem::graphTransform() const
{
    return m_graphTransform;
}

QRectF PlotItem::boundingRectFromData(const QList< double >& xData, const QList< double >& yData)
{
    int n = qMin(xData.size(), yData.size());
    if (n == 0)
        return QRectF();
    double x_min, x_max, y_min, y_max;
    x_min = x_max = xData[0];
    y_min = y_max = yData[0];
    for (int i = 1; i < n; ++i)
    {
        x_min = qMin(x_min, xData[i]);
        x_max = qMax(x_max, xData[i]);
        y_min = qMin(y_min, yData[i]);
        y_max = qMax(y_max, yData[i]);
    }
    return QRectF(x_min, y_min, x_max-x_min, y_max-y_min);
}

void PlotItem::setDataRect(const QRectF& dataRect) {
    m_dataRect = dataRect;
    if (m_graph)
    {
        m_graph->setDirty();
    }
}








