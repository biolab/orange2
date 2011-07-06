#include "plot.h"
#include "plotitem.h"

#include <QtCore/QDebug>

Plot::Plot(QWidget* parent): 
QGraphicsView(parent)
{
    setScene(new QGraphicsScene(this));
    clipItem = new QGraphicsRectItem();
    clipItem->setFlag(QGraphicsItem::ItemClipsChildrenToShape, true);
    scene()->addItem(clipItem);
    graph_item = new QGraphicsRectItem(clipItem);
}

Plot::~Plot()
{

}

void Plot::addItem(PlotItem* item)
{
    if (m_items.contains(item))
    {
        qWarning() << "Item is already in this graph";
        return;
    }
    item->m_plot = this;
    item->setParentItem(graph_item);
    m_items << item;
}

void Plot::removeItem(PlotItem* item)
{
    qDebug() << "Removing item" << item << "with parent" << item->parentItem();
    if (m_items.contains(item))
    {
        item->setParentItem(0);
        m_items.removeAll(item);
        item->m_plot = 0;
    }
    else
    {
        qWarning() << "Trying to remove an item that doesn't belong to this graph";
    }
}

QList< PlotItem* > Plot::itemList()
{
    qDebug() << "Returning itemlist with" << m_items.size() << "items";
    return m_items;
}

QRectF Plot::dataRectForAxes(int xAxis, int yAxis)
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

QPair< double, double > Plot::boundsForAxis(int axis)
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

void Plot::setDirty() 
{
    m_dirty = true;
}

void Plot::setClean() 
{
    m_dirty = false;
}

bool Plot::isDirty() 
{
    return m_dirty;
}

void Plot::setGraphRect(const QRectF rect) 
{
    clipItem->setRect(rect);
    graph_item->setRect(rect);
}

#include "plot.moc"
