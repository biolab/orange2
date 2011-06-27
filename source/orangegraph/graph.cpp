#include "graph.h"
#include "plotitem.h"

#include <QtCore/QDebug>

Graph::Graph(QWidget* parent): 
QGraphicsView(parent), 
graph_item(new QGraphicsRectItem())
{
    setScene(new QGraphicsScene(this));
    scene()->addItem(graph_item);
}

Graph::~Graph()
{

}

void Graph::addItem(PlotItem* item)
{
    item->m_graph = this;
    item->setParentItem(graph_item);
    m_items << item;
}

void Graph::removeItem(PlotItem* item)
{
    if (m_items.contains(item))
    {
        scene()->removeItem(item);
        m_items.removeAll(item);
    }
    else
    {
        qWarning() << "Trying to remove an item that doesn't belong to this graph";
    }
}

QList< PlotItem* > Graph::itemList()
{
    return m_items;
}

QRectF Graph::dataRectForAxes(int xAxis, int yAxis)
{
    QRectF r;
    QPair<int,int> axes = qMakePair(xAxis, yAxis);
    foreach (PlotItem* item, m_items)
    {
        if (item->isAutoScale() && item->axes() == axes)
        {
            r |= item->dataRect();
        }
    }
    return r;
}

QPair< double, double > Graph::boundsForAxis(int axis)
{
    
}
