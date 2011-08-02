#include "plot.h"
#include "plotitem.h"
#include "point.h"

#include <QtCore/QDebug>
#include <QtCore/qmath.h>
#include <limits>
#include "sceneeventfilter.h"

inline uint qHash(const DataPoint& pos)
{
    return pos.x + pos.y;
}

inline double distance(const QPointF& one, const QPointF& other)
{
    // For speed, we use the slightly wrong method, also known as Manhattan distance
    return (one - other).manhattanLength();
}

inline bool operator==(const DataPoint& one, const DataPoint& other)
{
    return one.x == other.x && one.y == other.y;
}

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
        Point* point = dynamic_cast<Point*>(item);
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
    scene()->installEventFilter(new SceneEventFilter(this));
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
    item->register_points();
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
    remove_all_points(item);
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
    if (behavior == ReplaceSelection)
    {
        unmark_all_points();
        behavior = AddSelection;
    }
    set_points_state(rect, scene(), Point::Marked, behavior);
}

void Plot::mark_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        unmark_all_points();
        behavior = AddSelection;
    }
    set_points_state(area, scene(), Point::Marked, behavior);
}

void Plot::select_points(const QRectF& rect, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        unselect_all_points();
        behavior = AddSelection;
    }
    set_points_state(rect, scene(), Point::Selected, behavior);
}

void Plot::select_points(const QPolygonF& area, Plot::SelectionBehavior behavior)
{
    if (behavior == ReplaceSelection)
    {
        unselect_all_points();
        behavior = AddSelection;
    }
    set_points_state(area, scene(), Point::Selected, behavior);
}

QList< int > Plot::selected_points(const QList< double > x_data, const QList< double > y_data, const QTransform& transform)
{
    Q_ASSERT(x_data.size() == y_data.size());
    const int n = qMin(x_data.size(), y_data.size());
    QList<int> selected;
    selected.reserve(n);
    DataPoint p;
    for (int i = 0; i < n; ++i)
    {
        p.x = x_data[i];
        p.y = y_data[i];
        selected << (selected_point_at(p) ? 1 : 0);
    }
    return selected;
}

Point* Plot::selected_point_at(const DataPoint& pos)
{
    foreach (PlotItem* item, plot_items())
    {
        if (m_point_set.contains(item) && m_point_set[item].contains(pos) && m_point_hash[item][pos]->is_selected())
        {
            return m_point_hash[item][pos];
        }
    }
    return 0;
}

Point* Plot::point_at(const DataPoint& pos)
{
    foreach (PlotItem* item, plot_items())
    {
        if (m_point_set.contains(item) && m_point_set[item].contains(pos))
        {
            return m_point_hash[item][pos];
        }
    }
    return 0;
}

Point* Plot::nearest_point(const QPointF& pos)
{
    QPair<double, DataPoint> closest_point = qMakePair( std::numeric_limits<double>::max(), DataPoint() );
    foreach (PlotItem* item, plot_items())
    {
        if (!m_point_set.contains(item))
        {
            continue;
        }
        PointSet::ConstIterator it = m_point_set[item].constBegin();
        PointSet::ConstIterator end = m_point_set[item].constEnd();
        for (it; it != end; ++it)
        {
            const double d = distance(m_point_hash[item][*it]->pos(), pos);
            if (d < closest_point.first)
            {
                closest_point.first = d;
                closest_point.second = *it;
            }
        }
    }
    Point* point = point_at(closest_point.second);
    if(distance(point->pos(), pos) <= point->size())
    {
        return point;
    }
    else
    {
        return 0;
    }
}

void Plot::add_point(const DataPoint& pos, Point* item, PlotItem* parent)
{
    m_point_set[parent].insert(pos);
    m_point_hash[parent].insert(pos, item);
}

void Plot::add_points(const Data& data, const QList< Point* >& items, PlotItem* parent)
{
    const int n = qMin(data.size(), items.size());
    for (int i = 0; i < n; ++i)
    {
        add_point(data[i], items[i], parent);
    }
}

void Plot::remove_point(const DataPoint& pos, PlotItem* parent)
{
    if (m_point_set.contains(parent) && m_point_set[parent].contains(pos))
    {
        m_point_set[parent].remove(pos);
        m_point_hash[parent].remove(pos);
    }
}

void Plot::remove_all_points(PlotItem* parent)
{
    if (m_point_set.contains(parent))
    {
        m_point_set[parent].clear();
        m_point_hash[parent].clear();
    }
}

void Plot::unmark_all_points()
{
    foreach (const PointHash& hash, m_point_hash)
    {
        foreach (Point* point, hash)
        {
            point->set_marked(false);
        }
    }
}

void Plot::unselect_all_points()
{
    int i = 0;
    foreach (const PointHash& hash, m_point_hash)
    {
        foreach (Point* point, hash)
        {
            ++i;
            point->set_selected(false);
        }
    }
    qDebug() << "Unselected" << i << "points";
}


#include "plot.moc"
