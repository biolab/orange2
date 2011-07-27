
#include "pointscollection.h"

#include "point.h"

uint qHash(const QPointF& pos)
{
    return pos.x() + pos.y();
}

PointsCollection::PointsCollection()
{

}

PointsCollection::~PointsCollection()
{

}

void PointsCollection::set_data_points(const QList< double >& x_data, const QList< double >& y_data)
{
    Q_ASSERT(x_data.size() == y_data.size());
    int n = qMin(x_data.size(), y_data.size());
    
    QList<QPointF> data;
    for (int i = 0; i < n; ++i)
    {
        data << QPointF(x_data[i], y_data[i]);
    }
    set_data_points(data);
}

void PointsCollection::set_data_points(const QList< QPointF >& data)
{
    m_data_points = data;
    m_data_indexes.clear();
    m_data_set.clear();
    const int n = m_data_points.size();
    for (int i = 0; i < n; ++i)
    {
        m_data_indexes.insert(m_data_points[i], i);
        m_data_set.insert(m_data_points[i]);
    }
}

void PointsCollection::set_points(const QList< Point* >& points)
{
    m_point_items = points;
}

Point* PointsCollection::point_at(const QPointF& pos) const
{
    if (!m_data_set.contains(pos))
    {
        return 0;
    }
    return m_point_items[m_data_indexes[pos]];
}

bool PointsCollection::contains(const QPointF& pos) const
{
    return m_data_set.contains(pos);
}

void PointsCollection::add_point(Point* point)
{
    m_point_items << point;
}

void PointsCollection::clear_points()
{
    m_point_items.clear();
}






