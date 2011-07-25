#include "plot.h"
#include "plotitem.h"

#include <QtCore/QDebug>

Plot::Plot(QWidget* parent): 
QGraphicsView(parent)
{
    setScene(new QGraphicsScene(this));
    clipItem = new QGraphicsRectItem();
    clipItem->setPen(Qt::NoPen);
    clipItem->setFlag(QGraphicsItem::ItemClipsChildrenToShape, true);
    scene()->addItem(clipItem);
    graph_item = new QGraphicsRectItem(clipItem);
}

Plot::~Plot()
{

}

void Plot::add_item(PlotItem* item)
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

void Plot::remove_item(PlotItem* item)
{
    if (m_items.contains(item))
    {
        item->setParentItem(0);
        m_items.removeAll(item);
        item->m_plot = 0;
        if (scene()->items().contains(item))
        {
            scene()->removeItem(item);
        }
    }
    else
    {
        qWarning() << "Trying to remove an item that doesn't belong to this graph";
    }
}

QList< PlotItem* > Plot::plot_items()
{
    return m_items;
}

QRectF Plot::data_rect_for_axes(int x_axis, int y_axis)
{
    QRectF r;
    QPair<int,int> axes = qMakePair(x_axis, y_axis);
    foreach (PlotItem* item, m_items)
    {
        if (item->is_auto_scale() && item->axes() == axes)
        {
            r |= item->data_rect();
        }
    }
    return r;
}

QPair< double, double > Plot::bounds_for_axis(int axis)
{
    QRectF y_r;
    QRectF x_r;
    foreach (PlotItem* item, m_items)
    {
        if (item->is_auto_scale())
        {
            if (item->axes().first == axis)
            {
               x_r |= item->data_rect(); 
            }
            else if (item->axes().second == axis)
            {
                y_r |= item->data_rect();
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

void Plot::set_dirty() 
{
    m_dirty = true;
}

void Plot::set_clean() 
{
    m_dirty = false;
}

bool Plot::is_dirty() 
{
    return m_dirty;
}

void Plot::set_graph_rect(const QRectF rect) 
{
    clipItem->setRect(rect);
    graph_item->setRect(rect);
}

#include "plot.moc"
