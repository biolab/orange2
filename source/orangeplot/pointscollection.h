#ifndef POINTSCOLLECTION_H
#define POINTSCOLLECTION_H

#include <QtCore/QList>
#include <QtCore/QPointF>
#include <QtCore/QHash>
#include <QtCore/QSet>

class Point;
class PointsCollection
{
public:
    PointsCollection();
    ~PointsCollection();
    
    Point* point_at(const QPointF& pos) const;
    inline Point* point_at(double x, double y) const
    {
        return point_at(QPointF(x, y));
    }
    bool contains(const QPointF& pos) const;
    inline bool contains(double x, double y) const
    {
        return contains(QPointF(x, y));
    }
    
protected:
    void set_points(const QList<Point*>& points);
    void add_point(Point* point);
    void clear_points();
    void set_data_points(const QList<double>& x_data, const QList<double>& y_data);
    void set_data_points(const QList<QPointF>& data);
    
private:
    QList<Point*> m_point_items;
    QList<QPointF> m_data_points;
    QSet<QPointF> m_data_set;
    QHash<QPointF, int> m_data_indexes;
};

#endif // POINTSCOLLECTION_H
