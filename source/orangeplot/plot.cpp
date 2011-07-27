#include "plot.h"
#include "plotitem.h"
#include "point.h"

#include <QtCore/QDebug>

template <class Area>
void set_points_state(Area area, QGraphicsScene* scene, Point::StateFlag flag, Plot::SelectionBehavior behavior)
{
    /*
     * NOTE: I think it's faster to rely on Qt to get all items in the current rect
     * than to iterate over all points on the graph and check which of them are 
     * inside the specified rect
     */
    foreach (QGraphicsItem* item, scene->items(area, Qt::IntersectsItemBoundingRect))
    {
        Point* point = qgraphicsitem_cast<Point*>(item);
        if (point)
        {
            point->set_state_flag(flag, behavior == Plot::AddSelection || (behavior == Plot::ToggleSelection && !point->state_flag(flag)));
        }
    }
}

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

void Plot::mark_points(const QRectF& rect, Plot::SelectionBehavior behavior)
{
    set_points_state(rect, scene(), Point::Marked, behavior);
}

void Plot::mark_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    set_points_state(area, scene(), Point::Marked, behavior);
}

void Plot::select_points(const QRectF& rect, Plot::SelectionBehavior behavior)
{
    set_points_state(rect, scene(), Point::Selected, behavior);
}

void Plot::select_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    set_points_state(area, scene(), Point::Selected, behavior);
}

QList< int > Plot::selected_points(const QList< double > x_data, const QList< double > y_data, const QTransform& transform)
{
    Q_ASSERT(x_data.size() == y_data.size());
    const int n = qMin(x_data.size(), y_data.size());
    QList<int> selected;
    selected.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        const QPointF coords = QPointF(x_data[i], y_data[i]) * transform;
        bool found_point = false;
        foreach (QGraphicsItem* item, scene()->items(coords))
        {
            if (item->pos() != coords)
            {
                continue;
            }
            const Point* point = qgraphicsitem_cast<Point*>(item);
            found_point = (point && point->is_selected());
            if (found_point)
            {
                break;
            }
        }
        selected << found_point;
    }
    qDebug() << "Found" << selected.count(1) << "selected points out of" << selected.size();
    return selected;
}

Point* Plot::point_at(const QPointF& pos)
{
    Point* point;
    foreach (QGraphicsItem* item, scene()->items(pos))
    {
        if (point = qgraphicsitem_cast<Point*>(item))
        {
            return point;
        }
    }
    return 0;
}

#include "plot.moc"
