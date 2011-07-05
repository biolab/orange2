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
    if (m_items.contains(item))
    {
        qWarning() << "Item is already in this graph";
        return;
    }
    item->m_graph = this;
    item->setParentItem(graph_item);
    m_items << item;
}

void Graph::removeItem(PlotItem* item)
{
    qDebug() << "Removing item" << item << "with parent" << item->parentItem();
    if (m_items.contains(item))
    {
        scene()->removeItem(item);
        m_items.removeAll(item);
        item->m_graph = 0;
    }
    else
    {
        qWarning() << "Trying to remove an item that doesn't belong to this graph";
    }
}

QList< PlotItem* > Graph::itemList()
{
    qDebug() << "Returning itemlist with" << m_items.size() << "items";
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
    QRectF y_r;
    QRectF x_r;
    foreach (PlotItem* item, m_items)
    {
        if (item->isAutoScale())
        {
            if (item->axes().first == axis)
            {
               x_r |= item->dataRect(); 
            }
            else if (item->axes().second == axis)
            {
                y_r |= item->dataRect();
            }
        }
    }
    if (x_r.isValid())
    {
        return qMakePair(x_r.left(), x_r.right());
    }
    else if (y_r.isValid())
    {
        return qMakePair(y_r.top(), y_r.bottom());
    }
    return qMakePair(0.0, 0.0);
}

void Graph::setDirty() 
{
    m_dirty = true;
}

void Graph::setClean() 
{
    m_dirty = false;
}

bool Graph::isDirty() 
{
    return m_dirty;
}

#include "graph.moc"
